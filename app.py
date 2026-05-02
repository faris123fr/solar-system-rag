import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. DOCUMENTS ---
DOCUMENTS = [
    "The Sun: At the center of our solar system lies the Sun, a nearly perfect sphere of hot plasma. It provides the energy necessary for life on Earth through nuclear fusion, turning hydrogen into helium deep within its core. This process releases a massive amount of energy in the form of light and heat, which takes about eight minutes to reach Earth. The Sun's gravity is what keeps every planet, from the smallest to the largest, locked in its orbit. Without the Sun, our solar system would be a dark, frozen void. Its surface temperature is approximately 5,500°C, while its core reaches a staggering 15 million degrees Celsius.",
    "Mercury: The smallest and innermost planet, Mercury is a world of extreme contrasts. Because it lacks a substantial atmosphere to trap heat, its surface temperatures can swing wildly. During the day, the Sun-facing side can reach a scorching 430°C, but at night, the temperature plummets to -180°C. Mercury is also the fastest planet, zipping around the Sun every 88 Earth days. Its surface is heavily cratered, resembling Earth's Moon, indicating that it has been geologically quiet for billions of years. Despite its proximity to the Sun, it is not the hottest planet; that title belongs to Venus.",
    "Venus: Often called Earth's twin due to its similar size and density, Venus is actually a hellish world. Its thick, toxic atmosphere is made mostly of carbon dioxide, with clouds of sulfuric acid. This atmosphere creates a runaway greenhouse effect, trapping solar heat and making Venus the hottest planet in our solar system with a constant surface temperature of about 465°C. This is hot enough to melt lead. Furthermore, the atmospheric pressure on the surface of Venus is 90 times higher than Earth's, which is equivalent to the pressure found 3,000 feet deep in Earth's oceans.",
    "Earth: Our home is the only known planet in the universe to support life. Earth is unique because of its liquid water oceans, which cover about 70% of the surface, and an atmosphere rich in nitrogen and oxygen. This 'Goldilocks' environment—not too hot and not too cold—is protected by a magnetic field that shields us from harmful solar radiation and cosmic rays. Earth is the only planet not named after a Greek or Roman deity; its name is an Old English word meaning 'the ground.' Its atmosphere is divided into five layers, with the troposphere being where all our weather occurs.",
    "Mars: Known as the Red Planet due to iron oxide (rust) on its surface, Mars is a cold, desert world with a very thin atmosphere. It is home to Olympus Mons, the largest volcano in the entire solar system, which is nearly three times the height of Mount Everest. Mars also features Valles Marineris, a canyon system that would stretch from New York to Los Angeles. Scientific evidence from rovers suggests that Mars once had liquid water on its surface billions of years ago. Today, water exists mainly as ice in the polar caps and beneath the Martian soil, making it a primary target for future human exploration.",
    "Jupiter: The king of planets, Jupiter is a gas giant made mostly of hydrogen and helium, much like a star. It is more than twice as massive as all the other planets in the solar system combined. Its most famous feature is the Great Red Spot, a massive storm larger than Earth that has been raging for at least 300 years. Jupiter has a powerful magnetic field and at least 95 moons, including Ganymede, the largest moon in the solar system. Because it is a gas giant, Jupiter does not have a solid surface; if you tried to stand on it, you would simply sink toward its incredibly hot and dense core.",
    "Saturn: Famous for its spectacular and complex ring system made of billions of chunks of ice and rock, Saturn is the second-largest planet. While all four gas giants have rings, Saturn's are the most visible and beautiful. Despite its size, Saturn is the least dense planet in the solar system; it is so light that it would actually float in water if you had a bathtub large enough. Saturn is also home to Titan, a moon with a thick atmosphere and lakes of liquid methane. The planet is mostly composed of hydrogen and helium, and it experiences intense winds that can reach speeds of 1,800 km/h.",
    "Uranus: An 'ice giant' with a unique tilt, Uranus rotates on its side at an angle of 98 degrees. This means its north and south poles face the Sun directly for long periods, leading to extreme 21-year-long seasons as it orbits the Sun every 84 Earth years. Uranus was the first planet discovered with a telescope, found by William Herschel in 1781. Its blue-green color is caused by methane gas in its cold atmosphere, which absorbs red light. Like Saturn, it has rings, though they are much thinner and darker. Uranus is also one of the coldest places in the solar system, with temperatures dropping to -224°C.",
    "Neptune: The most distant major planet, Neptune is a dark, cold, and windy world. Its winds are the fastest in the solar system, reaching speeds of over 2,000 km/h—fast enough to break the sound barrier. Its deep blue color comes from methane in its atmosphere, and it features a 'Great Dark Spot' similar to Jupiter's storm. Neptune was the first planet located through mathematical predictions rather than through regular observation. It has 14 known moons, the largest being Triton, which orbits the planet in the opposite direction of its rotation, suggesting it was once a captured dwarf planet from the Kuiper Belt.",
    "The Asteroid Belt and Dwarf Planets: Between the orbits of Mars and Jupiter lies the Asteroid Belt, a region filled with millions of rocky fragments left over from the early solar system. The largest object here is the dwarf planet Ceres. Far beyond Neptune lies the Kuiper Belt, a vast region of icy objects including the famous dwarf planet Pluto. Pluto was once considered the ninth planet but was reclassified in 2006 because it has not 'cleared its neighborhood' of other objects. These outer regions are like a cosmic time capsule, containing pristine materials that haven't changed since the planets first formed."
]

# --- 2. RAG LOGIC ---
@st.cache_resource
def setup_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.create_documents(DOCUMENTS)
    return Chroma.from_documents(chunks, embeddings)

vector_db = setup_db()

# --- 3. UI ---
st.title("🚀 Solar System Knowledge Base")
tab1, tab2 = st.tabs(["Search", "About"])

with tab1:
    query = st.text_input("What would you like to know?")
    if query:
        results = vector_db.similarity_search(query, k=3)
        for res in results:
            st.info(res.page_content)

with tab2:
    st.write("This RAG app allows semantic search through a Solar System database.")
    st.write("Built with Streamlit, LangChain, and ChromaDB.")
