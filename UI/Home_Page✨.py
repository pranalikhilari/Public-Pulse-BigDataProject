from turtle import position
from networkx import center
import streamlit as st
st.set_page_config(
    layout="wide",
    page_title="Public Pulse",
    page_icon="🧑‍💻",
    initial_sidebar_state='expanded'
    
)
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Call the function with the CSS file path
local_css("CSS/styles.css")
st.sidebar.success("Select a page above!")

import streamlit as st

def main():
    # Set page title and favicon
    st.markdown(
    """
    <style>
    .center {
        text-align: center;
    }
    </style>

    <div class="center">
        <h1>Welcome to Public Pulse!</h1>
    </div>
    """,
    unsafe_allow_html=True
    )
    st.markdown(
    """
    <style>
    @keyframes jump {
        0% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-20px);
        }
        100% {
            transform: translateY(0);
        }
    }
    .center {
        text-align: center;
    }
    .card {
        align
        animation: jump 3s ease infinite;
    }
    </style>

    <div class="card">
        <h4>&nbsp;&nbsp;&nbsp;Do you really know what people think of politics and politicians? Let's explore!</h4>
    </div>
    """,
    unsafe_allow_html=True
    )
    cola, colb, colc= st.columns([0.7,2,0.3])
    # Write the text inside the card
    with cola:
        pass
    with colb:
        # Add CSS styling for the card and image
        st.markdown(
        """
        <style>
        /* CSS for the card */
        .card {
            background-color: #00000;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
            
        }
        /* CSS for the image */
        .image {
            max-width: 100%; /* Maximum width of the image */
            height:5%; /* Auto adjust height to maintain aspect ratio */
            max-height: 10px;
            justify-content: center; /* Center horizontally */
        }
        </style>
        """,
        unsafe_allow_html=True
        )
        st.image("Output/HomePage.jfif", width=500)
    with colc:
        pass
    
    # Add a footer
    st.markdown(
        """
        <style>
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                height: 4   %;
                width: 100%;
                background-color: #333333;
                color: white;
                text-align: center;
                padding: 10px;
            }
        </style>
        <div class="footer">
            Made with ❤️ by PCube2.0
        </div>
        """,
        unsafe_allow_html=True
    )

# Run the main function
if __name__ == "__main__":
    main()

