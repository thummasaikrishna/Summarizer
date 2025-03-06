import nltk
import graphviz
import base64
from io import BytesIO
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import streamlit as st

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')

# Function to extract keywords
def extract_keywords(text, num_keywords=10):
    """Extract the most important keywords from text using TF-IDF."""
    stop_words = set(stopwords.words('english'))
    
    # Clean text
    text = ' '.join([word for word in text.lower().split() 
                    if word.isalnum() and word not in stop_words])
    
    # Use TF-IDF to extract important keywords
    vectorizer = TfidfVectorizer(max_features=num_keywords, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        keywords = vectorizer.get_feature_names_out()
        return keywords
    except:
        # Fallback if TF-IDF fails (e.g., with very short text)
        words = text.split()
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:num_keywords]]

# Function to build the mindmap structure
def build_mindmap_structure(keywords):
    """Create a hierarchical structure for the mindmap."""
    hierarchy = defaultdict(list)
    
    if len(keywords) == 0:
        root = "Mindmap"
    else:
        root = keywords[0]  # Set the most important keyword as the root
        
    main_topics = keywords[1:3] if len(keywords) > 2 else keywords[1:]
    sub_topics = keywords[3:] if len(keywords) > 3 else []
    
    for topic in main_topics:
        hierarchy[root].append(topic)
        
    if len(main_topics) == 2:
        left_subs = sub_topics[:len(sub_topics) // 2]
        right_subs = sub_topics[len(sub_topics) // 2:]
        
        for sub in left_subs:
            hierarchy[main_topics[0]].append(sub)
        for sub in right_subs:
            hierarchy[main_topics[1]].append(sub)
    
    return hierarchy, root

# Function to generate mindmap from text and return base64 encoded image
def generate_mindmap(text, theme="dark"):
    """Generate a mindmap visualization from text."""
    keywords = extract_keywords(text)
    hierarchy, root = build_mindmap_structure(keywords)
    
    # Configure graph based on theme
    if theme == "dark":
        bgcolor = "#121212"
        fontcolor = "white"
        root_color = "#6A5ACD"  # Slate blue
        main_topic_color = "#DAA520"  # Goldenrod
        subtopic_color = "#2E8B57"  # Sea green
        edge_color = "#FFFFFF"
    else:
        bgcolor = "#FFFFFF"
        fontcolor = "black"
        root_color = "#4B0082"  # Indigo
        main_topic_color = "#FF8C00"  # Dark orange
        subtopic_color = "#228B22"  # Forest green
        edge_color = "#000000"
    
    # Create the graph
    graph = graphviz.Digraph(format='png')
    graph.attr(bgcolor=bgcolor, fontcolor=fontcolor, rankdir="TB", 
               splines="curved", concentrate="true")
    
    # Add nodes and edges
    for parent, children in hierarchy.items():
        if parent == root:
            graph.node(parent, parent, shape='box', style='filled,rounded', 
                      fillcolor=root_color, fontcolor='white', fontsize="16")
        else:
            graph.node(parent, parent, shape='box', style='filled,rounded', 
                      fillcolor=main_topic_color, fontcolor='white')
            
        for child in children:
            graph.node(child, child, shape='box', style='filled,rounded', 
                      fillcolor=subtopic_color, fontcolor='white')
            graph.edge(parent, child, color=edge_color, penwidth="1.5")
    
    # Render the graph to a PNG image
    png_data = graph.pipe(format='png')
    
    # Encode to base64 for display in HTML
    encoded = base64.b64encode(png_data).decode('utf-8')
    
    return encoded

# UI section for the mindmap
def add_mindmap_section(summary_text, dark_mode=True, timestamp=None):
    """Add the mindmap UI section to the Streamlit app."""
    if summary_text:
        st.markdown("---")
        st.subheader("Key Concepts Mindmap")
        st.write("Visual representation of the key concepts in the summary.")
        
        # Use the current theme state
        theme = "dark" if dark_mode else "light"
        
        with st.spinner("Generating mindmap..."):
            try:
                encoded_image = generate_mindmap(summary_text, theme)
                st.markdown(f'<img src="data:image/png;base64,{encoded_image}" width="100%">', 
                           unsafe_allow_html=True)
                
                # Add download button for the mindmap image
                buffered = BytesIO(base64.b64decode(encoded_image))
                
                # Generate a timestamp-based filename if timestamp is provided
                file_name = f"mindmap_{timestamp}.png" if timestamp else "mindmap.png"
                
                st.download_button(
                    label="Download Mindmap",
                    data=buffered.getvalue(),
                    file_name=file_name,
                    mime="image/png",
                    key="mindmap_download"
                )
            except Exception as e:
                st.error(f"Failed to generate mindmap: {str(e)}")
                st.info("The mindmap generation requires text with sufficient content to identify key concepts.")
    else:
        st.info("Generate a summary first to see the mindmap visualization.")