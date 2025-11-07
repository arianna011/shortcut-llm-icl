from transformers.pipelines import TextGenerationPipeline
from .rep_control_reading_vec import WrappedReadingVecModel
import torch

class RepControlPipeline(TextGenerationPipeline):
    def __init__(self, 
                 model, 
                 tokenizer, 
                 layers, 
                 block_name="decoder_block", 
                 control_method="reading_vec",
                 **kwargs):
        
        # TODO: implement different control method and supported intermediate modules for different models
        assert control_method == "reading_vec", f"{control_method} not supported yet"
        assert block_name == "decoder_block" or "LlamaForCausalLM" in model.config.architectures, f"{model.config.architectures} {block_name} not supported yet"
        self.wrapped_model = WrappedReadingVecModel(model, tokenizer)
        self.wrapped_model.unwrap()
        self.wrapped_model.wrap_block(layers, block_name=block_name)
        self.block_name = block_name
        self.layers = layers

        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
   
    def __call__(self, text_inputs, activations=None, max_new_tokens=10, operator="linear_comb", **kwargs):

        if activations is not None:
            self.wrapped_model.reset()
            self.wrapped_model.set_controller(self.layers, activations, self.block_name, operator=operator)

        inputs = self.tokenizer(text_inputs, return_tensors="pt").to(self.model.device)

        # Run generation with score tracking
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        self.wrapped_model.reset()

        return {
            "sequences": output.sequences,
            "scores": output.scores                       # list of logits per token
        }
    
    # method that preserves gradients to allow interpretability techniques like Integrated Gradients
    def forward_with_grad(self, inputs_embeds, activations=None):
        self.wrapped_model.reset()
        if activations is not None:
            self.wrapped_model.set_controller(self.layers, activations, self.block_name)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            use_cache=False
        )
        self.wrapped_model.reset()
        return outputs