# from __future__ import annotations

# from typing import TYPE_CHECKING, Any, Dict, Optional

# from langchain_core.callbacks.base import BaseCallbackHandler

# if TYPE_CHECKING:
#     from langchain_core.agents import AgentAction, AgentFinish

# import streamlit as st



import streamlit as st
import pandas as pd
import io
import base64
import time
from crewai import Agent, Task, Crew
import os
import re
import sys
from datetime import datetime
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
#from langchain_community.utilities import SerpAPIWrapper #GoogleSerperAPIWrapper,
#from langchain.agents import Tool
#from crewai_tools import tool 

openai_api_key = st.secrets.secrets.OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = openai_api_key
serp_api_key = st.secrets.secrets.SERPER_API_KEY
os.environ["SERPER_API_KEY"] = serp_api_key
os.environ["OPENAI_MODEL_NAME"] = "gpt-4-turbo" #"gpt-4o" #'gpt-3.5-turbo'


# Initialize the tools
search_tool = SerperDevTool() #search #
scrape_tool = ScrapeWebsiteTool()

# @tool("GoogleSearch")
# def search(search_query: str):
#     """Search the web for information on a given topic"""
#     return SerpAPIWrapper().run(search_query)



#**************************************************************************************
# """Callback Handler that prints to std out."""




# class CustomStreamlitCallbackHandler(BaseCallbackHandler):
#     """Callback Handler that prints to std out."""

#     def __init__(self, color: Optional[str] = None) -> None:
#         """Initialize callback handler."""
#         self.color = color

#     def on_chain_start(
#         self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
#     ) -> None:
#         """Print out that we are entering a chain."""
#         class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
#         # print(f"\n\n\033[1m> Entering new {class_name} chain...\033[0m")  # noqa: T201
#         with st.expander("Starting a new Agent Chain:", expanded=True):
#             st.markdown(f"Entering new {class_name} chain...")

#     def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
#         """Print out that we finished a chain."""
#         # print("\n\033[1m> Finished chain.\033[0m")  # noqa: T201
#         with st.expander("Finished chain."):
#             st.write("Finished chain.")

#     def on_agent_action(
#         self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
#     ) -> Any:
#         """Run on agent action."""
#         # print_text(action.log, color=color or self.color)
#         with st.expander("AI Thought Bubble - Next Action:", expanded=True):
#             for line in action.log.split("\n"):
#                 st.markdown(line)

#     def on_tool_start(
#         self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
#     ) -> Any:
#         """Run when tool starts running."""
#         with st.expander("Tool Started", expanded=True):
#             st.write(serialized)
#             st.write(input_str)

#     def on_tool_end(
#         self,
#         output: str,
#         color: Optional[str] = None,
#         observation_prefix: Optional[str] = None,
#         llm_prefix: Optional[str] = None,
#         **kwargs: Any,
#     ) -> None:
#         """If not the final action, print out observation."""
#         with st.expander("Tool Ended:", expanded=True):
#             with st.expander("Obvervation:", expanded=True):
#                 if observation_prefix is not None:
#                     #     print_text(f"\n{observation_prefix}")
#                     st.markdown(f"\n{observation_prefix}")
#             st.markdown(f"\n{output}")
#             # print_text(output, color=color or self.color)
#             if llm_prefix is not None:
#                 with st.expander("LLM Prefix:", expanded=True):
#                     # print_text(f"\n{llm_prefix}")
#                     st.markdown(f"\n{llm_prefix}")

#     def on_text(
#         self,
#         text: str,
#         color: Optional[str] = None,
#         end: str = "",
#         **kwargs: Any,
#     ) -> None:
#         """Run when agent ends."""
#         # print_text(text, color=color or self.color, end=end)
#         with st.expander("Agent ending."):
#             st.write("Agent ending")

#     def on_agent_finish(
#         self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
#     ) -> None:
#         """Run on agent end."""
#         # print_text(finish.log, color=color or self.color, end="\n")
#         with st.expander("Agent Ended."):
#             st.write("Agent ended")
#***********************************************************************************



# #display the console processing on streamlit UI
# class StreamToExpander:
#     def __init__(self, expander):
#         self.expander = expander
#         self.buffer = []
#         self.colors = ['red', 'green', 'blue', 'orange']  # Define a list of colors
#         self.color_index = 0  # Initialize color index

#     def write(self, data):
#         # Filter out ANSI escape codes using a regular expression
#         cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

#         # Check if the data contains 'task' information
#         task_match_object = re.search(r'\"task\"\s*:\s*\"(.*?)\"', cleaned_data, re.IGNORECASE)
#         task_match_input = re.search(r'task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
#         task_value = None
#         if task_match_object:
#             task_value = task_match_object.group(1)
#         elif task_match_input:
#             task_value = task_match_input.group(1).strip()

#         if task_value:
#             st.toast(":robot_face: " + task_value)

#         # Check if the text contains the specified phrase and apply color
#         if "Entering new CrewAgentExecutor chain" in cleaned_data:
#             # Apply different color and switch color index
#             self.color_index = (self.color_index + 1) % len(self.colors)  # Increment color index and wrap around if necessary

#             cleaned_data = cleaned_data.replace("Entering new CrewAgentExecutor chain", f":{self.colors[self.color_index]}[Entering new CrewAgentExecutor chain]")

#         if "Website URL Finder" in cleaned_data:
#             # Apply different color 
#             cleaned_data = cleaned_data.replace("Website URL Finder", f":{self.colors[self.color_index]}[Website URL Finder]")
#         if "URL Verifier" in cleaned_data:
#             cleaned_data = cleaned_data.replace("URL Verifier", f":{self.colors[self.color_index]}[URL Verifier]")
#         #if "Technology Expert" in cleaned_data:
#           #  cleaned_data = cleaned_data.replace("Technology Expert", f":{self.colors[self.color_index]}[Technology Expert]")
#         if "Finished chain." in cleaned_data:
#             cleaned_data = cleaned_data.replace("Finished chain.", f":{self.colors[self.color_index]}[Finished chain.]")

#         self.buffer.append(cleaned_data)
#         if "\n" in data:
#             self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
#             self.buffer = []

def get_internal_links(url):
    try:
        # Send a GET request to the URL with a timeout
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Parse the URL to get the base domain
        base_domain = urlparse(url).netloc

        # List to store internal links
        internal_links = []

        # Find all anchor tags
        for link in soup.find_all('a', href=True):
            href = link['href']
            parsed_href = urlparse(href)
            
            # Check if the link is internal
            if parsed_href.netloc == base_domain or parsed_href.netloc == '':
                full_url = urljoin(url, href)
                if full_url not in internal_links:
                    internal_links.append(full_url)

        return internal_links
    
    except requests.exceptions.RequestException as e:
        print(f"Error accessing {url}: {e}")
        return [url]
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return [url]


# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Agent Emaitraction -  LLM Email Search",
    page_icon="logo_s.png",
    layout="wide",
    initial_sidebar_state="expanded",
    
)
# 450 450
st.markdown(
        """
       <style>
       [data-testid="stSidebar"][aria-expanded="true"]{
           min-width: 350px; 
           max-width: 350px;
       }
       """,
        unsafe_allow_html=True,
    )   

# Inject custom CSS for glowing border effect
st.markdown(
        """
        <style>
        .cover-glow {
            width: 70%;
            height: 70%;
            padding: 0px;
            box-shadow: 
                0 0 5px #330000,
                0 0 10px #660000,
                0 0 15px #990000,
                0 0 20px #CC0000,
                0 0 25px #FF0000,
                0 0 30px #FF3333,
                0 0 35px #FF6666;
            position: ;
            z-index: -1;
            border-radius: 30px;  /* Rounded corners */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Function to convert image to base64
def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load and display sidebar image with glowing effect
img_path = "logo_s.png"
img_base64 = img_to_base64(img_path)
st.sidebar.markdown(
        f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
        unsafe_allow_html=True,
    )
st.sidebar.markdown("---")

st.sidebar.markdown("""
            ## AI Agent Emaitraction
            
            Finding Specific Emails
            
            The AI Agent named, Emaitraction, aims to provide the specific email extraction service,
            and website finding from given list of school names and addresses,
            and it provides results in a tabular format for easier interpretation.
            Emaitraction uses state of the art LLMs to do extraction from webpages.
            
            Instructions for use:
            
            1. Upload School Names and Addresses data
            2. Proceed with "Run Website Search" to get websites for specific schools
            3. Auto Agent Search will provide Emails and their URLs
            4. Observe results and download
        """)

# Title of the Streamlit app
st.title("AI Agent For Finding Emails")
st.markdown("---")

@st.cache_resource
def website_crew(rows_as_strings, k):
    """This function will take school data as input and provide school official website by scraping the web via LLMs"""
    
    
    # Creating agents 
    linkfinder_agent = Agent(
        role="Website URL Finder",
        goal="To find the website url for given school name in a very efficient way",
        backstory=(
            "Your mission is to find website url for given school name and address data. " 
            "Find URL for this school: {schools} "
            "make sure to provide url for the given school name. "
            "Keep your responses and questions/inputs at minimum text to avoid rate limit error."
            "Only provide URL in your response."
        ),
        tools=[search_tool],
        allow_delegation = False,
        #verbose = True,
        max_rpm=30000,
        #callbacks=[CustomStreamlitCallbackHandler(color="white")],
    )
    
    linkverifier_agent = Agent(
        role="URL Verifier",
        goal="Your job is to counter verify URL for given School name and address data",
        backstory=(
            "Access and observe School Website and confirm that website belongs to the "
            "school data respectively. You will be provided the school name and its website."
            "Keep your responses and questions/inputs at minimum text to avoid rate limit error. "
            "Only provide URL in your response."
        ),
        tools=[scrape_tool],
        allow_delegation=True,
        #verbose = True,
        max_rpm=30000,
        #callbacks=[CustomStreamlitCallbackHandler(color="white")],
    )
    
    # Creating Tasks
    linkfinder_task = Task(
        description="Find the original website url of the given school name: {schools}",
        expected_output="The URL you found for given school along with school name and address. Only provide URL in your response.",
        #output_json=SchoolDetails,
        #output_file="school_details.json",
        agent=linkfinder_agent,
        
    )
    
    linkverifier_task = Task(
        description="Counter verify the found URL of given school name: {schools} "
                    "Check the link to confirm that it belongs to the school name respectively.",
        expected_output = "Verify URL and provide only URL in your response.",
        #output_file="confirm_urls.txt",
        agent=linkverifier_agent,
        
    )
    
    # Define the crew with agents and tasks
    url_extraction_crew = Crew(
        agents=[linkfinder_agent,
                linkverifier_agent
                ],
        tasks=[linkfinder_task,
            linkverifier_task
            ],
        #verbose=True
    )
    
    res = []
    res_st = ""
    for i in range(len(rows_as_strings[:k])):
        inputs = {"schools": rows_as_strings[i]}
        result = url_extraction_crew.kickoff(inputs=inputs)
        res.append(result)
        res_st += result
        if i % 10 == 0:
            #print("iteration stop: ", i)
            time.sleep(30)
        #print("Loop iteration#: ", i)
        
    return res, res_st

@st.cache_resource
def email_crew(web_list):
    """This function will take school websites as input and emails by scraping the web via LLMs"""
    
    emailfinder_agent = Agent(
        role="Official Email Finder",
        goal="As a School Official, your job is finding email of Parent-Teacher Group/PTO/PTA/PTSA/PTSO/PTC/PFC from given school's website links",
        backstory=(
            "Your mission is to find the email of pta (parent teacher association), pto (parent teacher organization), ptsa (parent teacher student association), ptso (parent teacher student organization), ptc, (parent teacher club), PFC (parent faculty club) or FPC, " 
            "from given official school website and its pages links: {school_website} "
            "These emails are most often found on the community or family section of the school website but you can check on given links. "
            "You are officially hired by School to perform science shows there and you need to find those emails "
            "so that the shows are scheduled for children as they're excited about science. Children will be sad if you do not find the emails."
            "Keep your responses and questions/inputs at minimum text to avoid rate limit error."
            "Only provide Email or contact form or facebook page and URL from where you got that email in your response separated by comma. if you don't find email then just return 'not found, not found'."
        ),
        tools=[scrape_tool],
        allow_delegation = False,
        verbose = True,
        max_rpm=30000,
    )

    emailconfirm_agent = Agent(
        role="Senior Email Verifier",
        goal="As a school official, you job is to counter verify that found email for given school website links is accurate",
        backstory=(
            "Access and observe school website links and confirm that pta (parent teacher association), pto (parent teacher organization), ptsa (parent teacher student association), ptso (parent teacher student organization), ptc, (parent teacher club), PFC (parent faculty club) or FPC Email is accurate, "
            "You will be provided official school website and its native links."
            "You are officially hired by School as researcher to verify those emails so that children's science shows can be scheduled on time "
            "Children will be sad if you do not verify and provide accurate emails from school website."
            "Keep your responses and questions/inputs at minimum text to avoid rate limit error."
            "Only provide Email or contact form or facebook page and URL from where you got that email in your response separated by comma. if you don't find email then just return 'not found, not found'."
        ),
        tools=[search_tool, scrape_tool],
        allow_delegation = True,
        verbose = True,
        max_rpm=30000,
    )
    
    emailfinder_task = Task(
        description="As a school official, your job is to find the pta (parent teacher association), pto (parent teacher organization), ptsa (parent teacher student association), ptso (parent teacher student organization), ptc (parent teacher club), PFC (parent faculty club) or FPC's Email of the given school website and its native links: {school_website}",
        expected_output="Only provide Email or contact form or facebook page and URL from where you got that email in your response separated by comma. if you don't find email then just return 'not found, not found'.",
        #output_json=SchoolDetails,
        #output_file="school_details.json",
        agent=emailfinder_agent,
        
    )

    emailverifier_task = Task(
        description="Counter verify the found pta (parent teacher association), pto (parent teacher organization), ptsa (parent teacher student association), ptso (parent teacher student organization), ptc (parent teacher club), PFC (parent faculty club) or FPC's Email for given school website and its native links: {school_website} ",
        expected_output = "Only provide Email or contact form or facebook page and URL from where you got that email in your response separated by comma. if you don't find email then just return 'not found, not found'.",
        #output_file="confirm_urls.txt",
        agent=emailconfirm_agent,
        
    )
    
    # Define the crew with agents and tasks
    email_extraction_crew = Crew(
        agents=[emailfinder_agent,
                emailconfirm_agent
                ],
        tasks=[emailfinder_task,
            emailverifier_task
            ],
        verbose=True
    )

    res_e = []
    res_st_e = ""
    for i in range(len(web_list)):
        internal_links = get_internal_links(web_list[i])
        inputs = {"school_website": internal_links}
        result = email_extraction_crew.kickoff(inputs=inputs)
        res_e.append(result)
        res_st_e += result
        if i % 10 == 0:
            print("iteration stop: ", i)
            time.sleep(30)
        print("Loop iteration#: ", i)
        
    return res_e, res_st_e

st.header("Upload Data of School Names and Addresses:")
# File uploader for Excel files
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
rows_as_strings = []
if uploaded_file is not None:
    # Read the Excel file into a DataFrame
    df = pd.read_excel(uploaded_file)

    # Display the DataFrame
    st.write("Original Excel file data:")
    st.dataframe(df)
    
    # Data used for website search
    st.write("School specific data for website search:")
    df_ = df[['School Name', 'District', 'County Name', 'Street Address', 'City']]
    st.dataframe(df_)

    
    # Specify the columns you want to extract
    coll_list = list(df[['School Name', 'District', 'County Name', 'Street Address', 'City']])
    columns_to_extract = coll_list #['School Name', 'District', 'County Name', 'Street Address', 'City']  # replace with your column names

    # Extract the specified columns
    selected_columns_df = df[columns_to_extract]

    # Convert the rows of the selected columns to a list of strings
    rows_as_strings = selected_columns_df.astype(str).values.tolist()

    # Concatenate the list elements into strings
    rows_as_strings = [' '.join(row) for row in rows_as_strings]

    k = 0
    res = None
    # df_["website"] = ""
    tot_ent = "Total schools: " + str(len(rows_as_strings))
    k = st.number_input("Specify number of School to process", value=0, placeholder=tot_ent)
    k = int(k)
    if st.button("Run Website Search"):
        # Placeholder for stopwatch
        stopwatch_placeholder = st.empty()
            
        # Start the stopwatch
        start_time = time.time()
        # with st.expander("Processing!"):
        #     #sys.stdout = StreamToExpander(st)
        #     with st.spinner("Generating Results"):
        res, res_st = website_crew(rows_as_strings, k)
        df_ = df_[:k]
        df_["website"] = res
        st.write("Found websites:")
        st.dataframe(df_)
        # Stop the stopwatch
        end_time = time.time()
        total_time = end_time - start_time
        stopwatch_placeholder.text(f"Total Time Elapsed: {total_time:.2f} seconds")
            
    
        if res is not None:

            st.header("Initiating Agent to Search")
            st.write("""Click "Start Searching" button so that agent will go through each website to find relevant emails.,""")

            web_list = df_["website"].tolist()

            #if st.button("Run Email Search"):
            # Placeholder for stopwatch
            stopwatch_placeholder_ = st.empty()
                    
            # Start the stopwatch
            start_time_ = time.time()
            # with st.expander("Processing!"):
            #     #sys.stdout = StreamToExpander(st)
            #     with st.spinner("Generating Results"):
            res_e, res_st_e = email_crew(web_list)
            # Initialize two empty lists
            emails = []
            urls = []

            # Loop through each item in the original list
            for item in res_e:
                # Split the string by comma and strip any extra whitespace
                parts = item.split(',')
                email = parts[0].strip()
                url = parts[1].strip()
                
                # Append the parts to their respective lists
                emails.append(email)
                urls.append(url)
            df_ = df_
            df_["email"] = emails
            df_["email_url"] = urls
            st.write("Found emails:")
            st.dataframe(df_)
            # Stop the stopwatch
            end_time_ = time.time()
            total_time_ = end_time_ - start_time_
            stopwatch_placeholder_.text(f"Total Time Elapsed: {total_time_:.2f} seconds")
            
            # # Convert the modified DataFrame back to an Excel file
            output = io.BytesIO()
            with pd.ExcelWriter(output) as writer:
                df_.to_excel(writer, index=False, sheet_name='Sheet1')
            processed_data = output.getvalue()

            # Generate the current date and time
            now = datetime.now()
            date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

            # Create the file name with the current date and time
            file_name = f"modified_file_{date_time_str}.xlsx"
            #Download button for the modified Excel file
            st.download_button(
                label="Download Excel file",
                data=processed_data,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
