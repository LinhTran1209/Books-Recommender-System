import plotly.express as px
import streamlit as st
import joblib


st.set_page_config(
    page_title="Visualization Chart",
    page_icon="📈",
    layout="wide",
)

if 'select_id_user' in st.session_state:
    st.sidebar.text(f"Id User: {st.session_state.select_id_user}")
else:
    st.sidebar.text("Id User: None")

if 'data_Full' not in st.session_state:
    st.session_state.data_Full = joblib.load('checkpoints/data_Full.pkl')

if 'name_books' in st.session_state and st.session_state.name_books is not None:
    st.session_state.name_books_page2 = st.session_state.name_books
    st.session_state.authors_page2 = st.session_state.authors
    st.session_state.years_page2 = st.session_state.years
    st.session_state.publishers_page2 = st.session_state.publishers
    st.session_state.img_books_page2 = st.session_state.img_books


col1, col2 =st.columns(2)

with col1:
    if 'name_books_page2' in st.session_state and st.session_state.name_books_page2 is not None:
        selected_book_page2 = st.selectbox('Name Book', st.session_state.name_books_page2)
        st.session_state.selected_book_page2 = selected_book_page2
        if st.session_state.selected_book_page2 is not None:
            if st.session_state.selected_book_page2 in st.session_state.name_books_page2:
                idx_book = st.session_state.name_books_page2.index(st.session_state.selected_book_page2)
                if 'img_books_page2' in st.session_state and len(st.session_state.img_books_page2) > idx_book:
                     # Chia col1 thành 2 phần (50% cho ảnh và 50% cho thông tin)
                    st.markdown(f"""
                    <div style="display: flex; width: 100%;">
                        <div style="width: 50%; padding: 10px 10px 0 0;">
                            <img src="{st.session_state.img_books_page2[idx_book]}" alt="{selected_book_page2}" style="width: 100%; height: auto; object-fit: cover; border-radius: 20px;"/>
                        </div>
                        <div style="width: 50%; padding: 10px;">
                            <h3>{selected_book_page2}</h3>
                            <p><strong>Author:</strong> {st.session_state.authors_page2[idx_book]}</p>
                            <p><strong>Year:</strong> {st.session_state.years_page2[idx_book]}</p>
                            <p><strong>Publisher:</strong> {st.session_state.publishers_page2[idx_book]}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


with col2:
    # print(st.session_state.data_Full)
    selected_attribute = st.selectbox('Select Attribute', ['Author', 'Year', 'Publisher'])

    filtered_data = st.session_state.data_Full[(st.session_state.data_Full['ID_User'] == st.session_state.select_id_user) & (st.session_state.data_Full['Rating'] > 6)]

    count_data = filtered_data[selected_attribute].value_counts().reset_index()
    count_data.columns = [selected_attribute, 'Count Rating']

    fig = px.bar(count_data, 
                 x='Count Rating', 
                 y=selected_attribute, 
                 orientation='h', 
                 title=f"Bar Chart for {selected_attribute} (Rating > 6)")
    
    fig.update_layout(
        title=dict(
            font=dict(size=23)
        ),
        height=600,
    )
    st.plotly_chart(fig)


st.markdown(f"""
    <div style="margin-top: 50px; display: flex; justify-content: center;">
        {st.session_state.data_Full[st.session_state.data_Full['ID_User'] == st.session_state.select_id_user][['ID_User', 'ID_Book', 'Name_Book', 'Author', 'Year', 'Publisher', 'Rating']].to_html(index=False)}
    </div>
""", unsafe_allow_html=True)


#https://www.youtube.com/watch?v=Sb0A9i6d320&list=PLHgX2IExbFovFg4DI0_b3EWyIGk-oGRzq