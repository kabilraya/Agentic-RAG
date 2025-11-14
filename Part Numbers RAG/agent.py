from retrieval import retrieve_part_numbers
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key: #if the api_key is not found raise an error 
    raise ValueError("API key not found")
genai.configure(api_key=api_key) #if found configure the generative_model with the api_key

def retrieval_of_partnumbers(base_query):
    """
    Retrieves the part numbers info according to the user query

    Args:
    base_query : The high level query provided by the user

    Returns:
        A list of retrieved docs from the vector database
    """

    retrieved_products = retrieve_part_numbers(base_query)
    for product in retrieved_products:
        print(product)
    print(len(retrieved_products))
    return retrieved_products

def chat(prompt:str):
    chat_model = genai.GenerativeModel(model_name="gemini-2.5-flash-lite",tools = [retrieval_of_partnumbers])

    chat_session = chat_model.start_chat()

    initial_prompt = f"""
    Your task is to help the user who can be a probable customer. The customer is very concious about the details of the products.
    The products in the databases are the mechanical parts which are the most detail focused.
    Your jobs or flow of work are as follows:
    1. When the user enters the query(base_query), you MUST call the tool "retrieval_of_partnumbers
    2. Use the list of products to answer the user query. 
    3. List all the retrieved products that is 30 products in a table with proper format ONLY IF the user doesn't ask about the specific product.
    4. Now for generating the specific answers go through the retrieved products. Go through the specifics such as dimensions of the products if the user has specified any.
    5. Also, for generating the specific answer identify the name of the product in the query. Convert every single characters in lowercase. And match the exact names and dimensions(specification). 
    6. Now, you MUST display the specific products in detail along with all the retrieved products. Also you can make some recommendation about similar products.
    6. Also MUST include the products' URL.
    User Query: "{prompt}"
    """

    response = chat_session.send_message(initial_prompt)

    try:
        function_call = response.candidates[0].content.parts[0].function_call
        if function_call is None:
            return response.text
        print(f"[AI] Decided to call Tool 1:{function_call.name}")
    except(IndexError,AttributeError,ValueError):
        print("\n[AI]: The model didn't call an functions and responded with text instead:")
        print(response.text)
        return response.text
    
    args = function_call.args
    tool_response = retrieval_of_partnumbers(base_query=args['base_query'])
    print(type(tool_response))
    response = chat_session.send_message({"function_response":{
        "name" : "retrieval_of_partnumbers",
        "response":{"document":tool_response}
    }})

    final_answer = response.candidates[0].content.parts[0].text
    return final_answer

def main():
    query = str(input("Enter your query:"))
    chat(query)

if __name__ == "__main__":
    main()