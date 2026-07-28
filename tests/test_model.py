import torch
from models import ConversationalGLM, GLMConfig
def test_forward_and_generation():
 config=GLMConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,max_position_embeddings=32)
 model=ConversationalGLM(config);ids=torch.tensor([[1,7,8,2]]);result=model(ids,labels=ids)
 assert result.logits.shape==(1,4,64);assert result.loss is not None and torch.isfinite(result.loss)
 generated=model.generate(ids[:,:2],max_new_tokens=3);assert generated.shape[1]>=2
