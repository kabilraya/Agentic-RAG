from retrieval import retrieve_relevant_documents
import google.generativeai as genai
# from google.generativeai.types import Part
from dotenv import load_dotenv
import os
import json
#Configuration of client and tools
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")   
if not api_key:
    raise ValueError("API KEY variable not set")
genai.configure(api_key=api_key)
def query_generation(base_query):
    """
    Generates diverse sub-queries based on a user's initial query.
    
    Args:
        base_query: The user's original,high level query.

    Returns:
        A list of generated sub-queries
    """

    generator_model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    prompt = f"""
    Based on the following user query, generate a list of 4 diverse search queries.
    Example: base_query: I want to purchase a Laptop of X Brand where X can be any brand(Acer,Lenovo etc)
    1. Price Comparision
    2. Specification Comparision
    3. Use Case (Gaming, Casual, Video-Editing)
    4. Comparison between X Brand and Y Brand on the same range
    5. Also include the original query in the list as well.
    base_query = {base_query}
    **Don't include any extra text just return the json format **
    Return only a JSON array of strings.
    """

    response = generator_model.generate_content(prompt)

    print("\n[DEBUG] Raw model output from query_generation:", response.text)

    try:
        sub_queries = json.loads(response.text)
    except json.JSONDecodeError:
        # Attempt to clean the output if the model wrapped JSON in text or markdown
        cleaned_text = response.text.strip("` \n").replace("json", "").strip()
        try:
            sub_queries = json.loads(cleaned_text)
        except json.JSONDecodeError:
            print("\n[ERROR] Model output is not valid JSON. Returning fallback list.")
            sub_queries = [base_query]  


    return {"query_list":sub_queries}


def deep_thinking_retrieval(query_list):
        """
        this function just gets the queries(user_queries) and subqueries and calls the retrieval function and loops through the queries
        
        Args:
            base_query: The original query from the user
            query_list = list of subqueries
        
        Returns:
            A single dictionary with key:"sub_query" and value:"the retrived docs"
        """
        documents_by_query = {}
        
        for query in query_list:
            document = retrieve_relevant_documents(query=query)
            
            documents_by_query[query] = document
        # print(json.dumps(documents_by_query, indent=2, ensure_ascii=False))
        return json.dumps(documents_by_query)

def chat(prompt:str):
    
    chat_model = genai.GenerativeModel(model_name="gemini-2.5-flash-lite", tools = [query_generation,deep_thinking_retrieval])
    
    
    
    chat_session = chat_model.start_chat()

    initial_prompt = f"""
    Your task is to help the user(a probable customer to find a laptop according to his/her need). Follow this instructions:
    1. First, call the "query_generation" tool wit the user's query.
    2. After you get the list of sub_queries, you MUST call the 'deep_thinking_retrieval' tool using that list as a parameter.
    3. Finally, use the retrieved documents to answer all the queries such as price comparision, specification comparision, X VS Y Laptop Brands where X is the laptop searched by the user, Use cases(Gaming, Light Weight use-case)
    4. Include all the possible answers analysing the retrieved docs and the all the queries generated. 
    5. Give a detailed answer that covers all the concerns of the user without him/her having to explain much in their query.
    6. Also, include the url for the laptop you mention in the response from the retrieved documents.
    7. Display the specification in a table for better readability. Also for difference between X and Y brand laptop. display it in a table.

    User Query: "{prompt}"
    """
    
    response = chat_session.send_message(initial_prompt)

    

    try:
        function_call = response.candidates[0].content.parts[0].function_call
        if function_call is None:
            raise ValueError("Model did not call a function")
        print(f"[AI] Decided to call Tool 1:{function_call.name}")
    except(IndexError,AttributeError):
        print("\n [AI] : The model didn't call a function and responded with text instead:")
        print(response.text)
        return


    args = function_call.args   
    tool_one_response_content = query_generation(base_query=args['base_query'])
    sub_queries = tool_one_response_content["query_list"]

    response = chat_session.send_message({
    "function_response": {
        "name": "query_generation",
        "response": {"query_list":sub_queries}
    }
    })

    try:
        function_call = response.candidates[0].content.parts[0].function_call
        if function_call is None:
            raise ValueError("Model did not call a function")
        print(f"[AI] Decided to call Tool 2:{function_call.name}")
    except(IndexError,AttributeError):
        print("\n [AI] : The model didn't call a function and responded with text instead:")
        print(response.text)
        return

    args = function_call.args
    
    tool_two_response_content = deep_thinking_retrieval(query_list=sub_queries)
    docs_dict = json.loads(tool_two_response_content)
    response = chat_session.send_message({"function_response":{
        "name":"deep_thinking_retrieval",
        "response": {"documents": docs_dict}
    }})
    final_answer = response.candidates[0].content.parts[0].text
    print(final_answer)
    return final_answer


    

