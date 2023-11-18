from hfppl import Model, LMContext, TokenCategorical, CachedCausalLM, smc_steer, smc_standard
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = np.array(['center', 'left', 'right'])
class2id = {
    'center': 0,
    'left': 1,
    'right': 2
}

### Example Usage ###
# Need to create a bias model wrapper first. This makes it so there's only one
# instance of the model, rather than loading it in every time we make a 
# prediction. Also need to load in the base LLM separately.

# We will use the TwistModel for steering, the NaiveModel is inefficient is 
# just for demonstration purposes.

# bias_model = bias_model_factory(path_to_bias_model, path_to_bias_tokenizer)
# llm = CachedCausalLM.from_pretrained(llm_model_name)

# summary = asyncio.run(gen_summary(llm_model_name, llm, bias_model, TwistModel, article, 'center'))


def bias_model_factory(bias_model_name, bias_tokenizer_name):
    bias_prediction_tokenizer = AutoTokenizer.from_pretrained(bias_tokenizer_name)

    bias_prediction_model = AutoModelForSequenceClassification.from_pretrained(bias_model_name)
    _ = bias_prediction_model.to(device)

    def bias_model(text):
        text_enc = bias_prediction_tokenizer(
            [text], truncation=True, padding=True, return_tensors='pt')

        outputs = bias_prediction_model(text_enc.input_ids.to(
            device), attention_mask=text_enc.attention_mask.to(device))
        logits = outputs.logits.detach().cpu()

        # Softmax makes more sense for single classifications
        sf_pred = outputs.logits.softmax(dim=-1).tolist()

        pred = classes[torch.tensor(sf_pred).numpy().argmax()]
        return pred, logits[0].numpy()
    return bias_model

class NaiveModel(Model):
  def __init__(self, lm, bias_model, prompt, target_bias, max_len=512):
    super().__init__()

    lm.cache_kv(lm.tokenizer.encode(prompt))

    self.context = LMContext(lm, prompt)

    self.prompt_len = len(str(self.context.s))

    self.target_bias = target_bias

    self.max_len = max_len

    self.bias_model = bias_model

  async def gen_sentence(self):
    token = await self.sample(self.context.next_token())
    self.max_len -= 1

    while True:
        yield token
        if str(token) in ['.', '!', '?'] or token.token_id == self.context.lm.tokenizer.eos_token_id or self.max_len <= 0:
            break
        token = await self.sample(self.context.next_token())
        self.max_len -= 1

  def condition_on_bias(self):
    summary = str(self.context.s)[self.prompt_len+1:]
    bias, _ = self.bias_model(summary)
    # print(bias)
    self.condition(bias == self.target_bias)

  async def step(self):
    sentence = []
    async for token in self.gen_sentence():
        sentence.append(token)

    # sentence_str = self.context.lm.tokenizer.decode(
    #     [t.token_id for t in sentence], skip_special_tokens=True)
    # print('next sentence:', sentence_str)

    self.condition_on_bias()

    if sentence[-1].token_id == self.context.lm.tokenizer.eos_token_id or self.max_len <= 0:
      self.finish()

  def immutable_properties(self):
      return set(['target_bias', 'prompt_len', 'bias_model'])

# everything else the same as the Naive Model
class TwistModel(NaiveModel):
    def condition_on_bias(self):
        summary = str(self.context.s)[self.prompt_len+1:]
        _, bias_logits = self.bias_model(summary)
        # print(bias_logits[class2id[self.target_bias]])
        self.twist(bias_logits[class2id[self.target_bias]])

async def gen_summary(llm_name, llm, bias_model, steer_model, article, target_bias):
    prompt = f'Summarize this article: {article}'

    if llm_name in ['gpt2']:
        print('Note: llm is gpt2 so prepend endoftext')
        prompt = f'<|endoftext|>{prompt}'

    model = steer_model(llm, bias_model, prompt, target_bias)

    particles = await smc_standard(model, 10)

    for i, p in enumerate(particles):
        print(f'Summary {i+1}:')

        summary = str(p.context.s)[p.prompt_len+1:-1]
        print(summary)

        pred_bias, _ = bias_model(summary)
        print(pred_bias)
        print(p.weight)
        print()

    weights = np.array([p.weight for p in particles])
    if weights.shape[0] == 0:
        print('No successful particles!')
        return ''

    best_particle = particles[weights.argmax()]
    return str(best_particle.context.s)[p.prompt_len+1:-1]