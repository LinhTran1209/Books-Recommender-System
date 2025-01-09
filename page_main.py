import joblib
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Book Recommendation",
    page_icon="💡",
)

if 'id_books' not in st.session_state:
    st.session_state.id_books = []
if 'name_books' not in st.session_state:
    st.session_state.name_books = []
if 'img_books' not in st.session_state:
    st.session_state.img_books = []
if 'authors' not in st.session_state:
    st.session_state.authors = []
if 'years' not in st.session_state:
    st.session_state.years = []
if 'publishers' not in st.session_state:
    st.session_state.publishers = []

# Định nghĩa lại model
def f_cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0
    
    similarity = dot_product / (norm_vector1 * norm_vector2)
    
    return similarity

class CF(object):
    def __init__(self, k, sim_func=f_cosine_similarity, mode="uu"):    
        self.k = k
        self.sim_func = sim_func
        self.mode = mode

        self.mean_ratings = None
        self.uti_matrix_Y = None
        self.simi_matrix_Y = None
        self.matrix_Y = None       
    
    def fit(self, matrix_Y):
        self.matrix_Y = matrix_Y
        self.n_users = matrix_Y.shape[1]  # số cột user
        self.n_items = matrix_Y.shape[0]   # số hàng item
        self.normalize()
        self.compute_similarity()

    def normalize(self):
        if self.mode == "uu":
            self.mean_ratings = np.nanmean(self.matrix_Y, axis=0)
            self.uti_matrix_Y = self.matrix_Y - self.mean_ratings 
        else:
            self.mean_ratings = np.nanmean(self.matrix_Y, axis=1)
            self.uti_matrix_Y = (self.matrix_Y.T - self.mean_ratings).T

        self.uti_matrix_Y = np.nan_to_num(self.uti_matrix_Y, nan=0)

    def compute_similarity(self):
        if self.mode == "uu":
            self.simi_matrix_Y = np.ones((self.n_users, self.n_users))
            for u in range(self.n_users):
                self.simi_matrix_Y[u, :] = [self.sim_func(self.uti_matrix_Y[:, u], self.uti_matrix_Y[:, j]) for j in range(self.n_users)]
        else:
            self.simi_matrix_Y = np.ones((self.n_items, self.n_items))     
            for i in range(self.n_items):
                self.simi_matrix_Y[i, :] = [self.sim_func(self.uti_matrix_Y[i, :], self.uti_matrix_Y[j, :]) for j in range(self.n_items)]

    def predict(self, u, i):
        if self.mode == "uu":
            users_rated_i = np.where(~np.isnan(self.matrix_Y[i, :]))[0]
            sim = self.simi_matrix_Y[u, users_rated_i]
            k_rated_sim = np.argsort(sim)[-self.k:]
            k_users_rated = users_rated_i[k_rated_sim]
            score_u = self.uti_matrix_Y[i, k_users_rated]
            score_i = sim[k_rated_sim]
        else:
            items_rated_u = np.where(~np.isnan(self.matrix_Y[:, u]))[0]
            sim = self.simi_matrix_Y[i, items_rated_u]
            k_rated_sim = np.argsort(sim)[-self.k:]
            k_items_rated = items_rated_u[k_rated_sim]
            score_u = self.uti_matrix_Y[k_items_rated, u]
            score_i = sim[k_rated_sim]
        
        denominator = np.sum(np.abs(score_i))
        if denominator != 0:
            pred_rating = np.sum(score_i * score_u) / denominator
        else:
            pred_rating = 0
        
        if self.mode == "uu":
            return pred_rating + self.mean_ratings[u] 
        else:
            return pred_rating + self.mean_ratings[i]

    def recommend(self, x):
        pred_ratings = []
        if self.mode == "uu":
            for i in range(self.n_items):
                if np.isnan(self.matrix_Y[i, x]):
                    pred_rating = self.predict(x, i)
                    pred_ratings.append((i, pred_rating))
        else:
            for u in range(self.n_users):
                if np.isnan(self.matrix_Y[x, u]):
                    pred_rating = self.predict(u, x)
                    pred_ratings.append((u, pred_rating))

        pred_ratings.sort(key=lambda x: x[1], reverse=True)
        top_pred_ratings = pred_ratings[:10]

        if self.mode == "uu":
            print(f"Các gợi ý cho user {x}:")
            for item, rating in top_pred_ratings:
                if rating > 5:
                    print(f"\t item {item} với giá trị dự đoán: {round(rating, 2)}")
        else:
            print(f"Các gợi ý cho item {x}:")
            for user, rating in top_pred_ratings:
                if rating > 5:
                    print(f"\t user {user} với giá trị dự đoán: {round(rating, 2)}")
        return top_pred_ratings

# Tìm kiếm theo id, index
def find_index_user(Utility_matrix, user_id):
    try:
        index = Utility_matrix.columns.get_loc(user_id)
        return index
    except KeyError:
        return f"User-ID {user_id} không tồn tại."
    
def find_id_book(Utility_matrix, index):
    try:
        title = Utility_matrix.index[index]
        return title
    except KeyError:
        return f"Index title {title} không tồn tại."

def recommend(user):
    idx_books = model.recommend(find_index_user(Utility_matrix, user))
    idx_books = [item[0] for item in idx_books]

    # Xóa danh sách trước đó
    st.session_state.id_books.clear()
    st.session_state.name_books.clear()
    st.session_state.img_books.clear()
    st.session_state.authors.clear()
    st.session_state.years.clear()
    st.session_state.publishers.clear()

    for i in range(len(idx_books)):
        st.session_state.id_books.append(find_id_book(Utility_matrix, idx_books[i]))

    for i in st.session_state.id_books:
        title = data_Full[data_Full['ID_Book'] == i]['Name_Book'].unique() 
        st.session_state.name_books.append(title[0])
        img_url = data_Full[data_Full['ID_Book'] == i]['Img_url'].unique() 
        st.session_state.img_books.append(img_url[0])
        author = data_Full[data_Full['ID_Book'] == i]['Author'].unique() 
        st.session_state.authors.append(author[0])
        year = data_Full[data_Full['ID_Book'] == i]['Year'].unique() 
        st.session_state.years.append(year[0])
        publisher = data_Full[data_Full['ID_Book'] == i]['Publisher'].unique() 
        st.session_state.publishers.append(publisher[0])

# Load lại các trọng số
model = CF(k=50, sim_func=f_cosine_similarity, mode="uu")
model = joblib.load('checkpoints/model.pkl')
data_Full = joblib.load('checkpoints/data_Full.pkl')
Utility_matrix = joblib.load('checkpoints/Utility_matrix.pkl')

st.title('Books Recommender System')
id_users = data_Full['ID_User'].unique()

if 'select_id_user' not in st.session_state:
    st.session_state.select_id_user = id_users[0]

st.session_state.select_id_user = st.selectbox('Id user', id_users, index=list(id_users).index(st.session_state.select_id_user))

if st.button('Show Recommendation'):
    recommend(st.session_state.select_id_user)

# Hiển thị các gợi ý nếu có
if st.session_state.name_books:
    for i in range(2):
        cols = st.columns(5) 
        for j in range(5):  
            index = i * 5 + j 
            if index < len(st.session_state.name_books):
                with cols[j]:
                    st.image(st.session_state.img_books[index])  
                    st.text(st.session_state.name_books[index])

    st.markdown(
        """
        <style>
        img {
            height: 200px; 
            object-fit: cover;  
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# streamlit run page_main.py