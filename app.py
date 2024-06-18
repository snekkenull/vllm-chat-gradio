import os
import gradio as gr
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model = os.environ.get("MODEL_ID")
model_name = model.split("/")[-1]

DESCRIPTION = f"""
<h3>MODEL: <a href="https://hf.co/{model}">{model_name}</a></h3>
<center>
<p>Qwen is the large language model built by Alibaba Cloud.
<br>
Feel free to test without log.
</p>
</center>
"""

css="""
h3 {
    text-align: center;
}
footer {
    visibility: hidden;
}
"""


# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model)



def generate(message, history, system, max_tokens, temperature, top_p, top_k, penalty):
    # Prepare your prompts
    conversation = [
        {"role": "system", "content":system}
    ]
    for prompt, answer in history:
        conversation.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}])
    conversation.append({"role": "user", "content": message})

    
    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    sampling_params = SamplingParams(
        temperature=temperature, 
        top_p=top_p, 
        top_k=top_k,
        repetition_penalty=penalty, 
        max_tokens=max_tokens,
        stop_token_ids=[151645,151643],
    )
    # generate outputs
    llm = LLM(model=model)
    outputs = llm.generate([text], sampling_params)
    
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        return generated_text


    
chatbot = gr.Chatbot(height=800)

with gr.Blocks(css=css) as demo:
    gr.HTML(DESCRIPTION)
    gr.ChatInterface(
        fn=generate,
        chatbot=chatbot,
        fill_height=True,
        additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
        additional_inputs=[
            gr.Textbox(value="You are a helpful assistant.", label="System message", render=False),
            gr.Slider(
                minimum=1, 
                maximum=30720, 
                value=2048, 
                step=1, 
                label="Max tokens",
                render=False,
            ),
            gr.Slider(
                minimum=0.1, 
                maximum=1.0, 
                value=0.7, 
                step=0.1, 
                label="Temperature",
                render=False,
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.95,
                step=0.05,
                label="Top-p",
                render=False,
            ),
            gr.Slider(
                minimum=0,
                maximum=20,
                value=20,
                step=1,
                label="Top-k",
                render=False,
            ),
            gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=1,
                step=0.1,
                label="Repetition penalty",
                render=False,
            ),
        ],
        retry_btn="Retry",
        undo_btn="Undo",
        clear_btn="Clear",
        submit_btn="Send",
    )

if __name__ == "__main__":
    demo.launch()