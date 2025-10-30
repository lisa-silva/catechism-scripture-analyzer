import streamlit as st
import requests
import json
import time
from typing import Dict, Any, List

# --- Configuration ---
# Read the API key from the standard Streamlit secrets configuration
# NOTE: If you are running this locally, you must have the .streamlit/secrets.toml file setup.
API_KEY = st.secrets.tool_auth.gemini_api_key
# Using a model known for strong reasoning and grounding
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
MODEL_NAME = "gemini-2.5-flash-preview-05-20"
MAX_RETRIES = 5

# --- Core LLM Function with Google Search Grounding ---

# Use st.cache_data to remember results for the same input, saving time and API calls
@st.cache_data(show_spinner=False)
def verify_claim(claim: str) -> Dict[str, Any]:
    """
    Sends a claim to the Gemini model with Google Search enabled to ground the response
    in external, verifiable information.
    """
    
    # 1. Define the NEW, highly specific System Prompt
    system_prompt = (
        "You are an impartial, highly detailed Scriptural Fact-Checker and Comparative Theologian. "
        "Your primary goal is to provide clarity by comparing Roman Catholic doctrine with explicit biblical support. "
        
        "Analyze the user's claim and structure your response into these two distinct, fact-based sections: "
        
        "1. **Roman Catholic Doctrine (Catechism/Tradition):** State the official Roman Catholic teaching regarding the claim. You MUST cite the Catechism of the Catholic Church (CCC) or official Magisterial tradition as the primary source for this doctrine. "
        
        "2. **Scriptural Verification (CSB, KJV, Other Bibles):** Examine the claim against the explicit text of the Bible, prioritizing the **Christian Standard Bible (CSB)** and **King James Version (KJV)** translations, and noting if other standard translations (like the New World Study Bible or World Study Bible) contain explicit references. Specifically note where direct, unambiguous scriptural support for the doctrine is **present or absent**. "
        
        "Use Google Search for grounding to ensure accuracy on both the Catechism text and the biblical textual status. Maintain a neutral, factual tone."
    )

    # 2. Define the User Query
    user_query = (
        f"Provide a comparative analysis of the claim: '{claim}'"
    )
    
    # 3. Construct the Payload
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "tools": [{"google_search": {} }], # Crucial: Enable Google Search for grounding
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    
    # ... (rest of the function for API calls, backoff, and source extraction) ...
    # (The rest of the function remains unchanged from the previous code)

    headers = {'Content-Type': 'application/json'}
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()
            
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            
            if candidate and candidate.get('content', {}).get('parts', [{}])[0].get('text'):
                text = candidate['content']['parts'][0]['text']
                sources = []
                grounding_metadata = candidate.get('groundingMetadata')

                if grounding_metadata and grounding_metadata.get('groundingAttributions'):
                    sources = grounding_metadata['groundingAttributions']
                    # Filter and extract URI and title from web attributions
                    sources = [
                        {'uri': attr['web']['uri'], 'title': attr['web']['title']}
                        for attr in sources if 'web' in attr and attr['web'].get('uri')
                    ]
                
                return {"text": text, "sources": sources}

            else:
                return {"text": "Error: Model returned an empty response candidate.", "sources": []}

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                # Exponential backoff
                delay = 2 ** attempt
                time.sleep(delay)
            else:
                return {"text": f"Error: Failed to connect to the verification service after {MAX_RETRIES} attempts. Details: {e}", "sources": []}
        except Exception as e:
            return {"text": f"An unexpected error occurred during API processing: {e}", "sources": []}


# --- Streamlit UI and Logic (Updated Title/Description) ---

def main():
    """Defines the layout and interactivity of the Streamlit app."""
    
    st.set_page_config(
        page_title="The Catechism-Scripture Analyzer", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.title("⚖️ The Catechism-Scripture Analyzer")
    st.markdown(
        """
        A neutral, fact-based tool designed to help you research and compare specific doctrines taught
        by the **Catechism of the Catholic Church (CCC)** against the explicit textual content of
        various Protestant Bibles, including the **Christian Standard Bible (CSB)** and the **King James Version (KJV)**.
        
        Enter any theological claim for a detailed, dual-source analysis.
        """
    )
    
    # Text Area for the user's claim
    claim_input = st.text_area(
        "Enter a Theological Claim for Comparative Analysis:",
        placeholder="E.g., 'The Catholic Church teaches Purgatory exists.' or 'The Assumption of Mary'",
        height=100
    )

    # Button to trigger the verification
    if st.button("Analyze Comparison", type="primary"):
        if claim_input:
            # Show a loading spinner while the API call is made
            with st.spinner("Searching official sources and performing dual-source analysis..."):
                results = verify_claim(claim_input)
            
            # --- Display Results ---
            st.markdown("### 🔎 Comparative Analysis Results")
            
            # Display the generated text
            st.markdown(results["text"])
            
            # Display the sources if they exist
            if results["sources"]:
                st.markdown("---")
                st.subheader("🌐 Grounding Sources")
                
                # Format sources nicely as a list
                source_list = ""
                for i, source in enumerate(results["sources"], 1):
                    # Ensure title is not empty, use URI as fallback
                    title = source.get('title') or source['uri']
                    source_list += f"- **[{title}]({source['uri']})**\n"
                
                st.markdown(source_list)
                st.caption("Note: Grounding sources are provided by Google Search and may include links to official Catechism documents or reputable theological sites.")
            else:
                st.warning("No specific grounding sources were found, or the model relied on internal knowledge.")
            
        else:
            st.warning("Please enter a claim to begin analysis.")

if __name__ == "__main__":
    main()
