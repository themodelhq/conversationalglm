from __future__ import annotations
from pathlib import Path
from typing import Iterable
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders
from transformers import PreTrainedTokenizerFast

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<image>", "<audio>", "<system>", "<user>", "<assistant>", "<tool>", "</tool>"]

class ConversationTokenizer:
    @staticmethod
    def train(texts: Iterable[str], output_dir: str | Path, vocab_size: int = 32000) -> PreTrainedTokenizerFast:
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tokenizer.train_from_iterator(texts, trainer=trainer)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output / "tokenizer.json"))
        fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<bos>", eos_token="<eos>", unk_token="<unk>", pad_token="<pad>", additional_special_tokens=SPECIAL_TOKENS[4:])
        fast.save_pretrained(output)
        return fast

    @staticmethod
    def load(path: str | Path) -> PreTrainedTokenizerFast:
        return PreTrainedTokenizerFast.from_pretrained(str(path))

def render_messages(messages: list[dict[str, str]], add_generation_prompt: bool = False) -> str:
    allowed = {"system", "user", "assistant", "tool"}
    turns = []
    for message in messages:
        role = message.get("role", "user")
        if role not in allowed:
            raise ValueError(f"Unsupported role: {role}")
        content = str(message.get("content", "")).strip()
        turns.append(f"<{role}>\n{content}\n")
    if add_generation_prompt:
        turns.append("<assistant>\n")
    return "".join(turns)
