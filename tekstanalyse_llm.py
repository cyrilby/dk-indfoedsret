# %% Test af brug af LLM til fx identificering af temaer [WIP as of 30-09-2024]

"""
=================
Kiril's comments:
=================
This approach may be promising to try, but the set-up seems
a bit finnicky and doesn't exactly deliver the expected results.
Follow up on this a bit later - preferably after you also know
how to use ChatGPT for the same purpose.


For the use of ChatGPT/Microsoft Copilot, which will be explored in a
separate script, visit this helpful tutorial:
https://www.browserbear.com/blog/using-chatgpt-api-in-python-5-examples/
"""

# Note: tænk over om det her skal flyttes til et særskilt skript

# # Import af model
# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # Tokenizer og model bliver hentet automatisk fra HuggingFace
# # Se mere her: https://huggingface.co/utter-project/EuroLLM-1.7B
# tokenizer = AutoTokenizer.from_pretrained("utter-project/EuroLLM-1.7B")
# model = AutoModelForCausalLM.from_pretrained(
#     "utter-project/EuroLLM-1.7B", ignore_mismatched_sizes=True
# )

# # Load the pre-trained model only once when the function is called the first time
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


# # Egen funktion til at inteagere med modellen
# def generate_text(prompt, max_length=100, num_return_sequences=1):
#     """
#     Generates text using a pre-trained LLM model.

#     Parameters:
#         prompt (str): The input text or question to prompt the model.
#         max_length (int): The maximum length of the generated text. Default is 100.
#         num_return_sequences (int): The number of generated responses. Default is 1.

#     Returns:
#         str: The generated text response.
#     """
#     response = pipe(
#         prompt, max_length=max_length, num_return_sequences=num_return_sequences
#     )
#     generated_text = response[0]["generated_text"]  # Extract the text from the response
#     return generated_text


# # Fx brug denne kommentar
# sample_text = all_debates["Udtalelse"].iloc[-12]
# generate_text("Venligst opsummer denne tekst:" + sample_text, 250, 1)
