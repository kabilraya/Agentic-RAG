from retrieval import retrieve_part_numbers
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import time

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
        print(product + "\n\n")
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
    2. Now you MUST use the retrieved documents to answer the user query.
    3. Determine what the user is asking:
    4. Only generate the answer in a table struture.
    5. The data stored in the vector db is as such:
        Part Number Name:C-CFUA3-10 | Price:107,596VND | 
        Days to Ship:Same Day | Minimum order Qty:1 Piece(s) | 
        Volumn Discount:Available | Outer Dia. D (mm):10 |
        Width B (mm):7 | Stud Screw Nominal (M) (mm):M3×0.5 | 
        Roller Guiding Method:With Retainer | Cam Follower: Stud Screw (Fine Thread) (mm):- |
        Basic Load Rating Cr (Dynamic) (kN):1.18 | 
        Basic Load Rating Cor (Static) (kN):0.94 | 
        Allowable Rotational Speed (rpm):26320 | 
        Seal:Provided | Application:Regular | 
        Category:camfollower | 
        Subcategory:cam followers crown hex socket | 
        URL: https://vn.misumi-ec.com/vona2/detail/110302715420/?HissuCode=C-CFUA3-10

    6. RULE FOR DETECTING AND SPLITTING TABLES BY SCHEMA (STRICT)
    You MUST execute the following algorithm before generating tables:

    STEP 1 — For each retrieved product:
    Extract all attribute names exactly as they appear (the text before the colon).

    STEP 2 — Build a SCHEMA SIGNATURE for each product:
    A Schema Signature = sorted list of all attribute names for that product.
    Example: ["Part Number Name", "Price", "Width B (mm)", "URL", ...]

    STEP 3 — Group products ONLY IF their Schema Signatures match exactly.
    - If two products have ANY difference in attribute names,
    even one missing or extra attribute,
    they belong to DIFFERENT groups.

    STEP 4 — Each group becomes ONE table.
    - Do NOT mix groups.
    - Split the groups as separate tables. Make it clearly separate.

    STEP 5 — Table generation rules:
    - Include ALL attributes from that schema as columns.
    - If a product has "-" or NULL for a value, KEEP the column and place "-" in the cell.
    - Do NOT reorder attributes.
    
    You must follow this algorithm EXACTLY.
    DO NOT merge the attributes of one group to another and make a huge table. Just Split the table having different schema.
    DO NOT mention which subcategory that table falls under. Just display SEPARATE TABLES IF THE SCHEMA IS DIFFERENT. Mention the sub category the table falls under.
    

    7. Provide the URL for each products.
    8. DO NOT ask any further question. Just generate responses according to the retrieved documnents matching the user's query against the rules above.
    User Query: "{prompt}"
    """
    llm_time_start = time.perf_counter()
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
    llm_time_end = time.perf_counter()
    llm_time = llm_time_end - llm_time_start
    with open("LLM_Generation_Time.txt","a",encoding="utf-8") as f:
        f.write(f"Query: {prompt} \n Time of Response Generation: {llm_time}\n\n")
    return final_answer

def main():
    query = str(input("Enter your query:"))
    chat(query)

if __name__ == "__main__":
    main()