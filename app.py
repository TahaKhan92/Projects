import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

# Load API key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if API key exists
if not GOOGLE_API_KEY:
    raise ValueError("Google API Key not found. Please set it in .env file.")

# Load AI Model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# AI Prompt for Book Recommendations
book_recommendation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant for a book recommendation platform. Recommend {count} books from the {category} category under the budget of {budget} with a rating of {rating}."),
        ("human", "Suggest {count} books from the {category} category with a rating of {rating} and under {budget}.")
    ]
)

# Format Response
format_response = RunnableLambda(lambda x: f"📚 Recommended Books:\n\n{x}")

# AI Chain
chain = book_recommendation_template | model | StrOutputParser() | format_response

# Function to get recommendations
def recommend_books(category, budget, count, rating):
    try:
        result = chain.invoke({"category": category, "budget": budget, "count": int(count), "rating": rating})
        return result
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Streamlit Interface
def main():
    # Title and description
    st.title("📚 Book Recommendation Bot")
    st.write(
        "Enter book category, budget, rating, and number of recommendations to get AI-based book suggestions."
    )

    # Inputs for category, budget, count, and rating
    category = st.text_input("Book Category", placeholder="e.g., Fiction, Science, Mystery")
    budget = st.text_input("Budget", placeholder="e.g., $10, $20, $50")
    rating = st.slider("Minimum Rating", 1, 5, 4, help="Select the minimum rating for the book (1 to 5 stars).")
    count = st.number_input("Number of Recommendations", min_value=1, value=3, step=1)

    # Get recommendations button
    if st.button("Get Recommendations"):
        if category and budget:
            # Get recommendations
            result = recommend_books(category, budget, count, rating)
            st.write(result)
        else:
            st.warning("Please fill in the product category and budget fields.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
