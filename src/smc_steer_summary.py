from hfppl import Model, LMContext, TokenCategorical, CachedCausalLM, smc_steer, smc_standard
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

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

# summary = await gen_summary(llm_model_name, llm, bias_model, TwistModel, article, 'center')


def bias_model_factory(bias_model_name, bias_tokenizer_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bias_prediction_tokenizer = AutoTokenizer.from_pretrained(bias_tokenizer_name)

    if torch.cuda.is_available():
        bias_prediction_model = AutoModelForSequenceClassification.from_pretrained(bias_model_name, device_map='auto')
    else:
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
  def __init__(self, lm_name, lm, bias_model, prompt, target_bias, max_len=512):
    super().__init__()

    # lm.cache_kv(lm.tokenizer.encode(prompt))

    self.context = LMContext(lm, prompt)
    self.lm_name = lm_name

    self.prompt_len = len(str(self.context.s))

    self.target_bias = target_bias

    self.max_len = max_len

    self.bias_model = bias_model

  def get_summary(self):
    return str(self.context.s)[self.prompt_len:]

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
    summary = self.get_summary()
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
      return set(['target_bias', 'prompt_len', 'bias_model', 'lm_name'])

# everything else the same as the Naive Model
class TwistModel(NaiveModel):
    def condition_on_bias(self):
        summary = self.get_summary()
        _, bias_logits = self.bias_model(summary)
        # print(bias_logits[class2id[self.target_bias]])
        self.twist(bias_logits[class2id[self.target_bias]])

async def gen_summary(llm_name, llm, bias_model, steer_model, article, target_bias):
    prompt = f'Summarize this article: {article}'
    prompt = llm.tokenizer.bos_token + prompt

    end_prompt = '\nSummary:'
    end_prompt_len = llm.tokenizer(end_prompt, return_tensors="pt").input_ids.shape[1]

    inputs = llm.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=llm.tokenizer.model_max_length - 512 - end_prompt_len)

    prompt = llm.tokenizer.decode(inputs.input_ids[0]) + end_prompt

    model = steer_model(llm_name, llm, bias_model, prompt, target_bias)

    particles = await smc_standard(model, 10)

    # for i, p in enumerate(particles):
    #     print(f'Summary {i+1}:')

    #     summary = p.get_summary()
    #     print(summary)

    #     pred_bias, _ = bias_model(summary)
    #     print(pred_bias)
    #     print(p.weight)
    #     print()

    weights = np.array([p.weight for p in particles])
    if weights.shape[0] == 0:
        print('No successful particles!')
        return ''

    best_particle = particles[weights.argmax()]
    return best_particle.get_summary()