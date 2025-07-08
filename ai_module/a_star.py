import streamlit as st
import folium
from streamlit_folium import st_folium
from queue import PriorityQueue
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(coord1, coord2):
    R = 6371  # Earth radius in kilometers
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


# ---------------- Coordinates ---------------- #
coordinates = {
    # Lahore
    'Lahore': (31.5497, 74.3436),
    'Mayo Hospital': (31.5784, 74.3192),
    'Shaukat Khanum': (31.4692, 74.4094),
    'Jinnah Hospital Lahore': (31.4934, 74.3009),
    'Services Hospital Lahore': (31.5546, 74.3162),
    'General Hospital Lahore': (31.5074, 74.3212),
    'Ittefaq Hospital Lahore': (31.4800, 74.3152),

    # Karachi
    'Karachi': (24.8607, 67.0011),
    'National Institute of Cardiovascular Diseases' : (24.8728, 67.0560),
    'Aga Khan Hospital': (24.8674, 67.0485),
    'Jinnah Hospital Karachi': (24.8714, 67.0737),
    'Ziauddin Hospital': (24.8810, 67.0439),
    'Indus Hospital Karachi': (24.8616, 67.0659),
    'Liaquat National Hospital': (24.8945, 67.0806),

    # Peshawar
    'Peshawar': (34.0151, 71.5805),
    'Hayatabad Medical Complex': (33.9987, 71.4425),
    'Lady Reading Hospital': (34.0076, 71.5482),
    'Khyber Teaching Hospital': (34.0006, 71.4995),

    # Quetta
    'Quetta': (30.1798, 66.9750),
    'Civil Hospital Quetta': (30.1843, 66.9987),
    'Sandeman Provincial Hospital': (30.1837, 66.9976),
    'Bolan Medical Complex Hospital': (30.1870, 67.0039),

    # Multan
    'Multan': (30.1575, 71.5249),
    'Nishtar Hospital': (30.2012, 71.4485),
    'Children Hospital Multan': (30.2101, 71.4545),
    'Multan Institute of Cardiology': (30.2041, 71.4569),

    # Rawalpindi/Islamabad
    'Rawalpindi': (33.6007, 73.0679),
    'Islamabad': (33.6844, 73.0479),
    'Blue Area': (33.7101, 73.0551),
    'Saddar': (33.5920, 73.0494),
    'Faizabad': (33.6581, 73.0715),
    'Chaklala': (33.5988, 73.0845),
    'Airport': (33.6177, 73.1007),
    'G-8': (33.6939, 73.0511),
    'G-10': (33.6982, 73.0252),
    'F-8': (33.7085, 73.0481),
    'I-8': (33.6770, 73.0906),
    'Bahria Town': (33.5335, 73.1198),
    'DHA Phase 2': (33.5256, 73.0949),
    'Holy Family Hospital': (33.6255, 73.0723),
    'Benazir Hospital': (33.6277, 73.0644),
    'CMH Rawalpindi': (33.5842, 73.0526),
    'Combined Military Hospital': (33.5896, 73.0804),
    'Rawalpindi Institute of Cardiology': (33.6253, 73.0702),
    'Polyclinic Hospital': (33.7176, 73.0712),
    'Ali Medical': (33.7012, 73.0244),
    'Shifa Hospital': (33.7076, 73.0574),
    'Maroof International': (33.7084, 73.0559),
    'KRL Hospital': (33.7165, 73.0925),
    'Islamabad Diagnostic Centre': (33.6752, 73.0937),
    'Riphah Hospital': (33.5257, 73.1278),
    'Mumtaz Hospital': (33.5281, 73.1301),
    'Al-Shifa Eye Hospital': (33.5215, 73.1003),
    'PIMS': (33.7100, 73.0500),
    'Federal Government Polyclinic': (33.7166, 73.0722),
    'Quaid-e-Azam International Hospital': (33.6560, 72.9877)
}

# ---------------- Graph ---------------- #
graph = {
    # Lahore and its areas
    'Lahore': {
        'Gulberg Lahore': {'distance': 3, 'road': 'Ferozepur Road'},
        'DHA Lahore': {'distance': 7, 'road': 'Canal Bank Road'},
        'Model Town Lahore': {'distance': 5, 'road': 'Multan Road'},
        'Johar Town Lahore': {'distance': 6, 'road': 'Shah Jamal Road'},
        'Shalimar Lahore': {'distance': 4, 'road': 'Ravi Road'}
    },
    'Gulberg Lahore': {
        'Mayo Hospital': {'distance': 2, 'road': 'Circular Road'},
        'Services Hospital Lahore': {'distance': 4, 'road': 'Jail Road'}
    },
    'DHA Lahore': {
        'Shaukat Khanum': {'distance': 5, 'road': 'Raiwind Road'},
        'General Hospital Lahore': {'distance': 8, 'road': 'Ferozepur Road'}
    },
    'Model Town Lahore': {
        'Ittefaq Hospital Lahore': {'distance': 3, 'road': 'Model Town Link Road'}
    },
    'Johar Town Lahore': {
        'Jinnah Hospital Lahore': {'distance': 4, 'road': 'Canal Bank Road'}
    },
    'Shalimar Lahore': {},
    'Mayo Hospital': {}, 'Shaukat Khanum': {}, 'Jinnah Hospital Lahore': {},
    'Services Hospital Lahore': {},
    'General Hospital Lahore': {},
    'Ittefaq Hospital Lahore': {},

    # Karachi and its areas
    'Karachi': {
        'Clifton Karachi': {'distance': 4, 'road': 'Stadium Road'},
        'Gulshan-e-Iqbal Karachi': {'distance': 7, 'road': 'University Road'},
        'Korangi Karachi': {'distance': 10, 'road': 'Korangi Road'},
        'Defense Karachi': {'distance': 6, 'road': 'Khayaban-e-Ittehad'},
        'Nazimabad Karachi': {'distance': 8, 'road': 'Shahrah-e-Faisal'}
    },
    'Clifton Karachi': {
        'Aga Khan Hospital': {'distance': 3, 'road': 'Stadium Road'},
        'Liaquat National Hospital': {'distance': 5, 'road': 'Stadium Road'}
    },
    'Gulshan-e-Iqbal Karachi': {
        'Ziauddin Hospital': {'distance': 6, 'road': 'Khayaban-e-Ghalib'},
        'National Institute of Cardiovascular Diseases': {'distance': 8, 'road': 'Abdullah Haroon Road'}
    },
    'Korangi Karachi': {
        'Indus Hospital Karachi': {'distance': 7, 'road': 'Korangi Road'}
    },
    'Defense Karachi': {},
    'Nazimabad Karachi': {},
    'Aga Khan Hospital': {}, 'Jinnah Hospital Karachi': {}, 'Ziauddin Hospital': {},
    'Indus Hospital Karachi': {}, 'Liaquat National Hospital': {},
    'National Institute of Cardiovascular Diseases': {},

    # Peshawar and its areas
    'Peshawar': {
        'Hayatabad Peshawar': {'distance': 6, 'road': 'University Road'},
        'University Town Peshawar': {'distance': 5, 'road': 'GT Road'},
        'Durrani Peshawar': {'distance': 4, 'road': 'Jamrud Road'},
        'Saddar Peshawar': {'distance': 7, 'road': 'Peshawar Road'},
        'Karkhano Peshawar': {'distance': 6, 'road': 'Saddar Road'}
    },
    'Hayatabad Peshawar': {
        'Hayatabad Medical Complex': {'distance': 2, 'road': 'University Road'}
    },
    'University Town Peshawar': {},
    'Durrani Peshawar': {},
    'Saddar Peshawar': {},
    'Karkhano Peshawar': {
        'Lady Reading Hospital': {'distance': 5, 'road': 'GT Road'}
    },
    'Hayatabad Medical Complex': {},
    'Lady Reading Hospital': {},
    'Khyber Teaching Hospital': {},

    # Quetta and its areas
    'Quetta': {
        'Satellite Town Quetta': {'distance': 3, 'road': 'Jinnah Road'},
        'Quetta Cantt': {'distance': 4, 'road': 'Cantt Road'},
        'Sariab Quetta': {'distance': 5, 'road': 'Sariab Road'},
        'Chaman Quetta': {'distance': 7, 'road': 'Chaman Road'},
        'Killi Quetta': {'distance': 6, 'road': 'Killi Road'}
    },
    'Satellite Town Quetta': {
        'Civil Hospital Quetta': {'distance': 2, 'road': 'Jinnah Road'}
    },
    'Quetta Cantt': {
        'Sandeman Provincial Hospital': {'distance': 3, 'road': 'Anscomb Road'}
    },
    'Sariab Quetta': {},
    'Chaman Quetta': {},
    'Killi Quetta': {
        'Bolan Medical Complex Hospital': {'distance': 5, 'road': 'BMC Road'}
    },
    'Civil Hospital Quetta': {},
    'Sandeman Provincial Hospital': {},
    'Bolan Medical Complex Hospital': {},

    # Multan (as before)
    'Multan': {
        'Nishtar Hospital': {'distance': 4, 'road': 'Nishtar Road'},
        'Children Hospital Multan': {'distance': 6, 'road': 'Abdali Road'},
        'Multan Institute of Cardiology': {'distance': 5, 'road': 'Qasim Bagh Road'}
    },
    'Children Hospital Multan': {},
    'Multan Institute of Cardiology': {},

    # Rawalpindi Main Areas
    'Rawalpindi': {
        'Saddar, Rawalpindi': {'distance': 4, 'road': 'Mall Road'},
        'Faizabad, Rawalpindi': {'distance': 8, 'road': 'Murree Road'},
        'Chaklala, Rawalpindi': {'distance': 6, 'road': 'Airport Road'},
        'Airport, Rawalpindi': {'distance': 20, 'road': 'Airport Road'}
    },
    'Saddar, Rawalpindi': {
        'Holy Family Hospital': {'distance': 2, 'road': 'Murree Road'},
        'Benazir Hospital': {'distance': 3, 'road': 'Benazir Bhutto Road'},
        'CMH Rawalpindi': {'distance': 4, 'road': 'The Mall'}
    },
    'Chaklala, Rawalpindi': {
        'Combined Military Hospital': {'distance': 5, 'road': 'GT Road'},
        'Rawalpindi Institute of Cardiology': {'distance': 5, 'road': 'Tipu Road'}
    },
    'Faizabad, Rawalpindi': {
        'Islamabad': {'distance': 6, 'road': 'Murree Road'},
        'Holy Family Hospital': {'distance': 5, 'road': 'Murree Road'}
    },
    'Airport, Rawalpindi': {
        'Bahria Town, Rawalpindi': {'distance': 15, 'road': 'GT Road'},
        'DHA Phase 2, Rawalpindi': {'distance': 12, 'road': 'Expressway'}
    },

    # Islamabad Main Areas
    'Islamabad': {
        'G-8, Islamabad': {'distance': 6, 'road': 'Jinnah Avenue'},
        'G-10, Islamabad': {'distance': 5, 'road': 'Kashmir Highway'},
        'F-8, Islamabad': {'distance': 6, 'road': 'Nazim-ud-din Road'},
        'Blue Area, Islamabad': {'distance': 5, 'road': 'Constitution Avenue'},
        'I-8, Islamabad': {'distance': 7, 'road': '8th Avenue'},
        'PIMS': {'distance': 4, 'road': 'Service Road South'},
        'Federal Government Polyclinic': {'distance': 3, 'road': 'Luqman Hakeem Road'},
        'Quaid-e-Azam International Hospital': {'distance': 14, 'road': 'GT Road'}
    },
    'G-8, Islamabad': {
        'Polyclinic Hospital': {'distance': 3, 'road': 'Luqman Hakeem Road'}
    },
    'G-10, Islamabad': {
        'Ali Medical': {'distance': 2, 'road': 'Khayaban-e-Iqbal'}
    },
    'F-8, Islamabad': {
        'Shifa Hospital': {'distance': 4, 'road': 'Park Road'},
        'Maroof International': {'distance': 2, 'road': 'College Road'}
    },
    'Blue Area, Islamabad': {
        'Polyclinic Hospital': {'distance': 2, 'road': 'Luqman Hakeem Road'},
        'KRL Hospital': {'distance': 6, 'road': 'Service Road East'}
    },
    'I-8, Islamabad': {
        'Islamabad Diagnostic Centre': {'distance': 2, 'road': 'Service Road North'}
    },

    # Surrounding areas
    'Bahria Town, Rawalpindi': {
        'Riphah Hospital': {'distance': 3, 'road': 'Main Boulevard'},
        'Mumtaz Hospital': {'distance': 2, 'road': 'Main Commercial Ave'}
    },
    'DHA Phase 2, Rawalpindi': {
        'Bahria Town, Rawalpindi': {'distance': 3, 'road': 'Expressway'},
        'Al-Shifa Eye Hospital': {'distance': 4, 'road': 'Defence Ave'}
    },

    # Terminal nodes (hospital endpoints)
    'Holy Family Hospital': {}, 'Benazir Hospital': {}, 'CMH Rawalpindi': {},
    'Combined Military Hospital': {}, 'Rawalpindi Institute of Cardiology': {},
    'Polyclinic Hospital': {}, 'Ali Medical': {}, 'Shifa Hospital': {},
    'Maroof International': {}, 'KRL Hospital': {}, 'Islamabad Diagnostic Centre': {},
    'Riphah Hospital': {}, 'Mumtaz Hospital': {}, 'Al-Shifa Eye Hospital': {},
    'PIMS': {}, 'Federal Government Polyclinic': {}, 'Quaid-e-Azam International Hospital': {}
}


# ---------------- Heuristics ---------------- #
heuristics = {
    # Rawalpindi/Islamabad
    'Rawalpindi': 10, 'Saddar': 6, 'Chaklala': 9, 'Faizabad': 7, 'Airport': 15,
    'Islamabad': 5, 'G-8': 4, 'G-10': 3, 'F-8': 3, 'Blue Area': 2, 'I-8': 3,
    'PIMS': 0, 'Bahria Town': 7, 'DHA Phase 2': 8,
    'Holy Family Hospital': 0, 'Benazir Hospital': 0,
    'CMH Rawalpindi': 0, 'Combined Military Hospital': 0,
    'Rawalpindi Institute of Cardiology': 0,
    'Polyclinic Hospital': 0, 'Ali Medical': 0,
    'Shifa Hospital': 0, 'Maroof International': 0,
    'KRL Hospital': 0, 'Islamabad Diagnostic Centre': 0,
    'Riphah Hospital': 0, 'Mumtaz Hospital': 0,
    'Al-Shifa Eye Hospital': 0,
    'Federal Government Polyclinic': 0,
    'Quaid-e-Azam International Hospital': 0,

    # Lahore
    'Lahore': 6,
    'Mayo Hospital': 0, 'Shaukat Khanum': 0, 'Jinnah Hospital Lahore': 0,
    'Services Hospital Lahore': 0, 'General Hospital Lahore': 0,
    'Ittefaq Hospital Lahore': 0,

    # Karachi
    'National Institute of Cardiovascular Diseases': 0,
    'Karachi': 15,
    'Aga Khan Hospital': 0, 'Jinnah Hospital Karachi': 0,
    'Ziauddin Hospital': 0, 'Indus Hospital Karachi': 0,
    'Liaquat National Hospital': 0,

    # Peshawar
    'Peshawar': 12,
    'Hayatabad Medical Complex': 0, 
    'Lady Reading Hospital': 0,
    'Khyber Teaching Hospital': 0,

    # Quetta
    'Quetta': 20,
    'Civil Hospital Quetta': 0, 
    'Sandeman Provincial Hospital': 0,
    'Bolan Medical Complex Hospital': 0,

    # Multan
    'Multan': 10,
    'Nishtar Hospital': 0, 
    'Children Hospital Multan': 0,
    'Multan Institute of Cardiology': 0
}

# ---------------- Mapping ---------------- #
hospital_specialists = {
    'Holy Family Hospital': 'General Physician',
    'Benazir Hospital': 'Cardiologist',
    'CMH Rawalpindi': 'Orthopedic',
    'Combined Military Hospital': 'General Physician',
    'Rawalpindi Institute of Cardiology': 'Cardiologist',
    'Polyclinic Hospital': 'General Physician',
    'Ali Medical': 'Dermatologist',
    'Shifa Hospital': 'Pulmonologist',
    'Maroof International': 'Endocrinologist',
    'KRL Hospital': 'Neurologist',
    'Islamabad Diagnostic Centre': 'Radiologist',
    'Riphah Hospital': 'Orthopedic',
    'Mumtaz Hospital': 'General Physician',
    'Al-Shifa Eye Hospital': 'Ophthalmologist',
    'PIMS': 'Endocrinologist',
    'Mayo Hospital': 'General Physician',
    'Shaukat Khanum': 'Oncologist',
    'Jinnah Hospital Lahore': 'Cardiologist',
    'Aga Khan Hospital': 'Neurologist',
    'Jinnah Hospital Karachi': 'General Physician',
    'Hayatabad Medical Complex': 'Orthopedic',
    'Civil Hospital Quetta': 'Dermatologist',
    'Nishtar Hospital': 'Cardiologist',
    'Services Hospital Lahore': 'General Physician',
    'General Hospital Lahore': 'Neurologist',
    'Ittefaq Hospital Lahore': 'Cardiologist',
    'Ziauddin Hospital': 'Oncologist',
    'Indus Hospital Karachi': 'Pulmonologist',
    'Liaquat National Hospital': 'Endocrinologist',
    'Lady Reading Hospital': 'General Physician',
    'Khyber Teaching Hospital': 'Orthopedic',
    'Sandeman Provincial Hospital': 'Cardiologist',
    'Bolan Medical Complex Hospital': 'Neurologist',
    'Children Hospital Multan': 'Pediatrician',
    'Multan Institute of Cardiology': 'Cardiologist',
    'Federal Government Polyclinic': 'General Physician',
    'Quaid-e-Azam International Hospital': 'General Physician',
    'National Institute of Cardiovascular Diseases' : 'Cardiologist'
}


disease_specialty = {
    'asthma': 'Pulmonologist',
    'diabetes': 'Endocrinologist',
    'flu': 'General Physician',
    'skin allergy': 'Dermatologist',
    'heart attack': 'Cardiologist',
    'bone fracture': 'Orthopedic',
    'eczema': 'Dermatologist',
    'thyroid': 'Endocrinologist',
    'fever': 'General Physician',
    'migraine': 'Neurologist',
    'eye infection': 'Ophthalmologist',
    'high blood pressure': 'Cardiologist',
    'broken leg': 'Orthopedic'
}

# ---------------- A* Algorithm ---------------- #
def a_star_search(start, goal):
    visited = set()
    pq = PriorityQueue()
    # Priority queue stores tuples: (f_cost, g_cost, current_node, path)
    pq.put((0 + heuristics.get(start, 0), 0, start, [start]))

    while not pq.empty():
        f_cost, g_cost, current, path = pq.get()

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return path, g_cost

        for neighbor, data in graph.get(current, {}).items():
            if neighbor not in visited:
                new_g_cost = g_cost + data['distance']
                # f_cost = g_cost + heuristic estimate to goal
                new_f_cost = new_g_cost + heuristics.get(neighbor, 0)
                pq.put((new_f_cost, new_g_cost, neighbor, path + [neighbor]))

    return None, float('inf')  # if no path found

def get_coord(loc):
    # Try direct match
    if loc in coordinates:
        return coordinates[loc]
    # Try to match by removing common suffixes (like ", Rawalpindi", ", Islamabad")
    for key in coordinates:
        if loc.lower().startswith(key.lower()):
            return coordinates[key]
        if key.lower() in loc.lower():
            return coordinates[key]
    return None

# -------------------------------- #
def find_nearest_hospital(start, specialist):
    best_path = None
    best_cost = float('inf')
    best_hospital = None

    # Only consider hospitals with the required specialist
    candidate_hospitals = [h for h, s in hospital_specialists.items() if s == specialist]

    # Try to find a hospital with the required specialist using the graph
    for hospital in candidate_hospitals:
        path, cost = a_star_search(start, hospital)
        if path is not None and cost < best_cost:
            best_path = path
            best_cost = cost
            best_hospital = hospital

    # If no path found, fallback: find the geographically nearest hospital with the specialist
    if best_path is None or best_path == []:
        min_geo_dist = float('inf')
        for hospital in candidate_hospitals:
            start_coord = get_coord(start)
            hosp_coord = get_coord(hospital)
            if start_coord and hosp_coord:
                geo_dist = haversine_distance(start_coord, hosp_coord)
                if geo_dist < min_geo_dist:
                    min_geo_dist = geo_dist
                    best_path = [start, hospital]
                    best_cost = round(geo_dist, 2)
                    best_hospital = hospital

    # Always return a hospital with the required specialist, even if far away
    if best_path:
        return best_path, best_cost, True
    else:
        return None, float('inf'), False
    
    
    
"""    
# ---------------- Streamlit UI ---------------- #
st.set_page_config(page_title="🩺 MediAssist", page_icon="🚑")
if "result" not in st.session_state:
    st.session_state.result = None
if "path" not in st.session_state:
    st.session_state.path = None
if "total_distance" not in st.session_state:
    st.session_state.total_distance = None

st.title("🩺 MediAssist: Hospital Finder")

st.markdown("Select a disease and your current location to find the nearest hospital with a relevant specialist using the A* algorithm.")

selected_disease = st.selectbox("Select your disease:", sorted(disease_specialty.keys()))
# Use all locations from coordinates as possible starting points, but exclude hospitals
hospital_names = set(hospital_specialists.keys())

# Add main areas for each city in the format "Area, City"
main_areas = [
    # Lahore
    "Gulberg Lahore", "DHA Lahore", "Model Town Lahore", "Johar Town Lahore", "Shalimar Lahore",
    # Karachi
    "Clifton Karachi", "Gulshan-e-Iqbal Karachi", "Korangi Karachi", "Defense Karachi", "Nazimabad Karachi",
    # Peshawar
    "Hayatabad Peshawar", "University Town Peshawar", "Durrani Peshawar", "Saddar Peshawar", "Karkhano Peshawar",
    # Quetta
    "Satellite Town Quetta", "Quetta Cantt", "Sariab Quetta", "Chaman Quetta", "Killi Quetta",
    # Multan
    # (If you have areas for Multan, add here)
    # Rawalpindi (with and without city suffix)
    "Saddar, Rawalpindi", "Faizabad, Rawalpindi", "Chaklala, Rawalpindi", "Airport, Rawalpindi", "Bahria Town, Rawalpindi", "DHA Phase 2, Rawalpindi",
    "Saddar", "Faizabad", "Chaklala", "Airport", "Bahria Town", "DHA Phase 2",
    # Islamabad (with and without city suffix)
    "G-8, Islamabad", "G-10, Islamabad", "F-8, Islamabad", "Blue Area, Islamabad", "I-8, Islamabad",
    "G-8", "G-10", "F-8", "Blue Area", "I-8"
]

# Only include main areas that are present in coordinates and not hospitals
hospital_names = set(hospital_specialists.keys())
city_locations = sorted([loc for loc in coordinates.keys() if loc not in hospital_names])
start_location = st.selectbox("Select your starting location:", city_locations)

# 🔘 Button logic to search and store in session_state
if st.button("Find Nearest Hospital"):
    required_specialist = disease_specialty[selected_disease]
    path, total_distance, specialist_found = find_nearest_hospital(start_location, required_specialist)

    if path:
        if specialist_found:
            st.session_state.result = f"✅ Found a {required_specialist} at **{path[-1]}**!"
        else:
            st.session_state.result = f"⚠️ No hospital with a {required_specialist} was reachable. Showing nearest available hospital at **{path[-1]}** instead."
        
        st.session_state.path = path
        st.session_state.total_distance = total_distance

        
    else:
        st.session_state.result = "❌ No hospital found from your starting location."
        st.session_state.path = None
        st.session_state.total_distance = None


# ✅ Display the results outside the button logic
if st.session_state.result:
    if st.session_state.path:
        st.success(st.session_state.result)
        # ...existing code...

        st.markdown("### 🗺️ Route:")

        for i in range(len(st.session_state.path) - 1):
            road = graph.get(st.session_state.path[i], {}).get(st.session_state.path[i+1], {}).get("road", "")
            if road and road != "Unknown Road":
                st.write(f"{i+1}. {st.session_state.path[i]} → {st.session_state.path[i+1]} via *{road}*")
            else:
                st.write(f"{i+1}. {st.session_state.path[i]} → {st.session_state.path[i+1]}")
        st.write(f"{len(st.session_state.path)}. Destination: **{st.session_state.path[-1]}**")

        st.markdown(f"📏 **Total distance:** `{st.session_state.total_distance} km`")

        st.markdown("### 🗺️ Map View:")
        center_coords = coordinates.get(start_location, (33.6007, 73.0679))  # Default fallback
        m = folium.Map(location=center_coords, zoom_start=12, tiles="CartoDB Positron")

        points = []
        for idx, loc in enumerate(st.session_state.path):
            coord = get_coord(loc)
            if coord:
                points.append(coord)
                # Mark start in green, hospital in red, others in blue
                if idx == 0:
                    color = 'green'
                    popup = f"Start: {loc}"
                elif idx == len(st.session_state.path) - 1:
                    color = 'red'
                    popup = f"Hospital: {loc}"
                else:
                    color = 'blue'
                    popup = loc
                folium.Marker(
                    location=coord,
                    popup=popup,
                    icon=folium.Icon(color=color)
                ).add_to(m)

        # Remove the PolyLine to avoid showing a straight line
        # if len(points) >= 2:
        #     folium.PolyLine(points, color='blue', weight=5, opacity=0.7).add_to(m)

        st_folium(m, width=700, height=500)

# ...existing code...


    else:
        st.error(st.session_state.result)
        
"""