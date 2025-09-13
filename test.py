# %%
# %load_ext autoreload
# %autoreload 2

# %%
import time
from datetime import datetime
from random import Random

import torch
import torch.optim as optim
from torch import Tensor

# from torch.utils.tensorboard import SummaryWriter

from spargel_llm.data_source import PlainTextSource
from spargel_llm.tokenizer import UnicodeTokenizer
from spargel_llm.v1.torch import LLM, calculate_loss

# %%
torch.set_num_threads(16)
torch.set_printoptions(linewidth=200)

# writer = SummaryWriter("runs/llm_" + datetime.now().isoformat())

# %%
max_seq_len = 64

text_file = "nietzsche.txt"

# %%
with open(text_file) as f:
    text = f.read()

print("text length:", len(text))

random = Random()

text_source = PlainTextSource(text, max_seq_len + 1, max_seq_len + 1, random=random)

reserved_vocab = ["<|pad|>", "<|unknown|>"]
PAD, UNKNOWN = range(0, len(reserved_vocab))

tokenizer = UnicodeTokenizer.train_from_text(text, reserved_vocab, unknown=UNKNOWN)

print("vocab_size:", tokenizer.vocab_size)

print("example texts:")
for _ in range(16):
    print("  " + repr(text_source.sample()))


# %%
def prepare_data(batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
    inputs, masks, targets = [], [], []
    for text in text_source.sample_multiple(batch_size):
        tokens = tokenizer.encode(text)
        input_tokens = tokens[:max_seq_len]
        target_tokens = tokens[1 : max_seq_len + 1]
        input_tokens += (max_seq_len - len(input_tokens)) * [PAD]
        target_tokens += (max_seq_len - len(target_tokens)) * [PAD]

        cut_pos = max_seq_len
        # cut_pos = random.randint(max_seq_len // 2, max_seq_len)
        for i in range(cut_pos, max_seq_len):
            input_tokens[i] = PAD
            target_tokens[i] = PAD

        inputs.append(torch.tensor(input_tokens))
        targets.append(torch.tensor(target_tokens))

        # padding mask
        mask = torch.zeros(max_seq_len, dtype=torch.bool)
        mask[cut_pos:] = True
        masks.append(mask)

    return torch.stack(inputs), torch.stack(masks), torch.stack(targets)


example_inputs, example_masks, example_targets = prepare_data(16)
print("example inputs:")
for input in example_inputs:
    print("  " + repr(tokenizer.decode(input.tolist())))

# %%
# torch._dynamo.config.compiled_autograd = True


@torch.compile
def train_step(
    model: LLM,
    optimizer: optim.Optimizer,
    inputs: Tensor,
    masks: Tensor,
    targets: Tensor,
):
    optimizer.zero_grad()
    loss = calculate_loss(model, inputs, masks, targets, pad_index=PAD)
    loss.backward()
    optimizer.step()


class TrainState:
    steps: int
    time: float

    def __init__(self, steps: int = 0, time: float = 0.0):
        self.steps = steps
        self.time = time


def train(
    state: TrainState,
    model: LLM,
    optimizer: optim.Optimizer,
    batch_size: int,
    batches: int,
    epochs: int,
):
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")

        start_t = last_t = time.perf_counter()
        batch_delta_t = 0

        last_state_time = state.time

        for i in range(batches):
            # prepare data
            inputs, masks, targets = prepare_data(batch_size)

            # train
            model.train()

            train_step(model, optimizer, inputs, masks, targets)

            state.steps += 1

            t = time.perf_counter()
            delta_t = t - last_t
            last_t = t

            batch_delta_t += delta_t
            state.time += delta_t

            # val
            if i % 100 == 99:

                model.eval()

                with torch.no_grad():
                    train_inputs, train_masks, tarin_targets = prepare_data(200)
                    val_inputs, val_masks, val_targets = prepare_data(200)

                    train_loss = calculate_loss(
                        model,
                        train_inputs,
                        train_masks,
                        tarin_targets,
                        pad_index=PAD,
                    ).item()
                    val_loss = calculate_loss(
                        model, val_inputs, val_masks, val_targets, pad_index=PAD
                    ).item()

                print(
                    f"  Step {i+1:04d}/{batches:04d} {batch_delta_t:10.6f} s: train loss {train_loss:10.6f}, val loss {val_loss:10.6f}"
                )

                # writer.add_scalar("loss/train", train_loss, state.steps)
                # writer.add_scalar("loss/val", val_loss, state.steps)
                # writer.flush()

                batch_delta_t = 0

        print(
            f"  Epoch time: {state.time - last_state_time:10.6f} s, total time: {state.time:.6f} s"
        )

    print("Done.")


# %%
dim = 16
model = LLM(
    tokenizer.vocab_size,
    max_seq_len,
    cnt_h=2,
    dim=dim,
    d_key=dim,
    d_value=dim,
    d_hidden=dim,
)

# writer.add_graph(model, (example_inputs, example_masks))
# writer.flush()

train_state = TrainState()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
train(train_state, model, optimizer, 16, 1000, 10)


# %%
@torch.compile
def generate_step(model: LLM, input: Tensor) -> Tensor:
    return model(input)


def generate(model: LLM, text: str, max_new_tokens: int, temperature: float):

    tokens = tokenizer.encode(text)

    model.eval()

    for _ in range(max_new_tokens):
        input = tokens[-max_seq_len:]

        with torch.no_grad():
            logits = generate_step(model, torch.tensor(input))  # (seq_len, vocab_size)

        logits = logits[-1, :]  # (vocab_size)
        probs = torch.softmax(logits / temperature, dim=-1)
        next = int(torch.multinomial(probs, num_samples=1).item())

        tokens.append(next)

        print(tokenizer.decode([next]), end="", flush=True)

    return tokens


print("generated text:")
tokens = generate(model, "The", 1000, 0.5)

# # %%
# loss = calculate_loss(
#     model, example_inputs, example_masks, example_targets, pad_index=PAD
# )
# print("loss on examples:", loss.item())

# # %%
# for param in model.parameters():
#     print("================")
#     print(param.shape)
#     print(param)

# # %%
# batch_size = torch.export.Dim("batch_size")
# seq_len = torch.export.Dim("seq_len")
# dynamic_shapes = {
#     "tokens": {0: batch_size, 1: seq_len},
#     "mask": {0: batch_size, 1: seq_len},
# }

# model.train()
# torch.onnx.export(
#     model,
#     (example_inputs, example_masks),
#     "llm.onnx",
#     export_params=True,
#     input_names=["tokens", "mask"],
#     output_names=["logits"],
#     dynamic_shapes=dynamic_shapes,
#     dynamo=True,
#     training=torch.onnx.TrainingMode.TRAINING,
# )
