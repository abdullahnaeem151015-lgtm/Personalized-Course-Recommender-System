import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
)
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

# =========================================================
# Page setup
# =========================================================
st.set_page_config(
    page_title="Personalized Learning Recommender",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1240px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #F9FAFB 0%, #EFF6FF 100%);
        border: 1px solid #DBEAFE;
        margin-bottom: 1rem;
    }
    .hero h1 {
        color: #45425A;
        margin: 0;
        font-size: 2.15rem;
    }
    .hero p {
        color: #4B5563;
        margin: .45rem 0 0;
        max-width: 820px;
    }
    .guide {
        padding: 1rem 1.1rem;
        border: 1px solid #DBEAFE;
        border-radius: 14px;
        background: #F8FBFF;
        margin-bottom: 1rem;
    }
    .soft-card {
        padding: 1rem 1.1rem;
        border-radius: 16px;
        border: 1px solid #E5E7EB;
        background: white;
        min-height: 130px;
    }
    .soft-card h4 {
        margin: 0 0 .4rem;
        color: #2563EB;
    }
    .step {
        padding: .9rem 1rem;
        border-left: 4px solid #2563EB;
        background: #F9FAFB;
        border-radius: 0 12px 12px 0;
        margin-bottom: .7rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: .75rem;
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Personalized Learning Recommender</h1>
      <p>
        Choose courses you have completed, compare recommendation methods,
        and understand why each course appears in your result.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Public datasets used by the completed recommender workflow
# =========================================================
RATINGS_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBMSkillsNetwork-ML0321EN-Coursera/labs/v2/module_3/ratings.csv"
)
COURSE_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-ML321EN-SkillsNetwork/labs/datasets/course_processed.csv"
)
COURSE_GENRE_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv"
)
USER_PROFILE_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-ML321EN-SkillsNetwork/labs/datasets/user_profile.csv"
)
BOW_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-ML321EN-SkillsNetwork/labs/datasets/courses_bows.csv"
)
SIM_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-ML321EN-SkillsNetwork/labs/datasets/sim.csv"
)
USER_EMBED_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-ML321EN-SkillsNetwork/labs/datasets/user_embeddings.csv"
)
COURSE_EMBED_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "IBM-ML321EN-SkillsNetwork/labs/datasets/course_embeddings.csv"
)


@st.cache_data(show_spinner=False, ttl=3600)
def read_csv(url):
    return pd.read_csv(url)


@st.cache_data(show_spinner="Loading course data...")
def load_data():
    return {
        "ratings": read_csv(RATINGS_URL),
        "courses": read_csv(COURSE_URL),
        "genres": read_csv(COURSE_GENRE_URL),
        "profiles": read_csv(USER_PROFILE_URL),
        "bows": read_csv(BOW_URL),
        "similarity": read_csv(SIM_URL),
        "user_embeddings": read_csv(USER_EMBED_URL),
        "course_embeddings": read_csv(COURSE_EMBED_URL),
    }


try:
    DATA = load_data()
except Exception as exc:
    st.error(
        "The course data could not be loaded. Check the internet connection "
        "and refresh the app."
    )
    st.exception(exc)
    st.stop()

ratings = DATA["ratings"].copy()
courses = DATA["courses"].copy()
genres = DATA["genres"].copy()
profiles = DATA["profiles"].copy()
bows = DATA["bows"].copy()
sim_df = DATA["similarity"].copy()
user_embeddings = DATA["user_embeddings"].copy()
course_embeddings = DATA["course_embeddings"].copy()

course_lookup = courses[
    ["COURSE_ID", "TITLE", "DESCRIPTION"]
].drop_duplicates("COURSE_ID")


# =========================================================
# Shared helpers
# =========================================================
def minmax_score_map(score_map):
    """Normalize method-specific scores only for easier within-list comparison."""
    if not score_map:
        return {}

    keys = list(score_map)
    values = np.asarray([score_map[k] for k in keys], dtype=float)
    finite = np.isfinite(values)

    if not finite.any():
        return {k: 0.0 for k in keys}

    valid_values = values[finite]
    low = float(valid_values.min())
    high = float(valid_values.max())

    if np.isclose(low, high):
        return {k: 1.0 for k in keys}

    return {
        key: float((value - low) / (high - low)) if np.isfinite(value) else 0.0
        for key, value in score_map.items()
    }


def confidence_label(match_strength):
    if match_strength >= 80:
        return "Strong relative match"
    if match_strength >= 60:
        return "Good relative match"
    if match_strength >= 40:
        return "Possible match"
    return "Explore with caution"


def generic_reason(method):
    reasons = {
        "Course Similarity": "Similar to the content of the courses you selected.",
        "User Profile": "Matches the course-genre interests found in your selected courses.",
        "Clustering": "Popular among learners grouped with a similar profile.",
        "Clustering with PCA": "Popular in a similar learner group after reducing profile features.",
        "KNN": "Has an interaction pattern close to the courses you selected.",
        "NMF": "Received a higher predicted score from the factorized user-item matrix.",
        "Neural Network Embeddings": "Its learned course vector aligns with the selected learner vector.",
    }
    return reasons.get(method, "Ranked by the selected recommendation method.")


def build_results(
    score_map,
    selected_ids,
    top_n,
    method,
    reason_map=None,
    secondary_map=None,
):
    if not score_map:
        return pd.DataFrame(
            columns=[
                "RANK",
                "COURSE_ID",
                "TITLE",
                "DESCRIPTION",
                "RAW_SCORE",
                "MATCH_STRENGTH",
                "CONFIDENCE",
                "WHY_RECOMMENDED",
            ]
        )

    normalized = minmax_score_map(score_map)
    rows = []

    for course_id, raw_score in score_map.items():
        if course_id in selected_ids:
            continue

        reason = (
            reason_map.get(course_id)
            if reason_map and course_id in reason_map
            else generic_reason(method)
        )

        if secondary_map and course_id in secondary_map:
            reason = f"{reason} {secondary_map[course_id]}"

        rows.append(
            {
                "COURSE_ID": course_id,
                "RAW_SCORE": float(raw_score),
                "MATCH_STRENGTH": round(normalized.get(course_id, 0.0) * 100, 1),
                "WHY_RECOMMENDED": reason,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result = result.merge(course_lookup, on="COURSE_ID", how="left")
    result = (
        result.sort_values(
            ["MATCH_STRENGTH", "RAW_SCORE"], ascending=False
        )
        .drop_duplicates("COURSE_ID")
        .head(top_n)
        .reset_index(drop=True)
    )
    result.insert(0, "RANK", np.arange(1, len(result) + 1))
    result["CONFIDENCE"] = result["MATCH_STRENGTH"].apply(confidence_label)

    return result[
        [
            "RANK",
            "COURSE_ID",
            "TITLE",
            "DESCRIPTION",
            "RAW_SCORE",
            "MATCH_STRENGTH",
            "CONFIDENCE",
            "WHY_RECOMMENDED",
        ]
    ]


def selected_course_genre_profile(selected_ids):
    genre_columns = [c for c in genres.columns if c not in ["COURSE_ID", "TITLE"]]
    selected = genres[genres["COURSE_ID"].isin(selected_ids)]
    if selected.empty:
        return np.zeros(len(genre_columns)), genre_columns
    return selected[genre_columns].sum(axis=0).to_numpy(dtype=float), genre_columns


@st.cache_resource(show_spinner=False)
def prepare_similarity_map():
    grouped = bows.groupby(["doc_index", "doc_id"]).max().reset_index(drop=False)
    idx_to_id = grouped[["doc_id"]].to_dict()["doc_id"]
    id_to_idx = {course_id: idx for idx, course_id in idx_to_id.items()}
    return id_to_idx, sim_df.to_numpy(dtype=float)


@st.cache_resource(show_spinner="Preparing the learner clusters...")
def train_clusters(use_pca, n_clusters):
    feature_columns = [c for c in profiles.columns if c != "user"]
    x = profiles[feature_columns].to_numpy(dtype=float)

    if use_pca:
        full_pca = PCA()
        full_pca.fit(x)
        cumulative = np.cumsum(full_pca.explained_variance_ratio_)
        component_count = int(np.argmax(cumulative >= 0.90) + 1)
        reducer = PCA(n_components=component_count, random_state=42)
        model_x = reducer.fit_transform(x)
    else:
        reducer = None
        component_count = len(feature_columns)
        model_x = x

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(model_x)

    labelled = ratings.merge(
        pd.DataFrame({"user": profiles["user"], "cluster": labels}),
        on="user",
        how="inner",
    )
    popularity = (
        labelled.groupby(["cluster", "item"])["user"]
        .nunique()
        .reset_index(name="enrollments")
    )
    return model, reducer, feature_columns, popularity, component_count


@st.cache_resource(show_spinner="Preparing the user-course matrix...")
def interaction_matrix():
    return ratings.pivot_table(
        index="user",
        columns="item",
        values="rating",
        fill_value=0,
    )


@st.cache_resource(show_spinner="Training the NMF representation...")
def train_nmf(n_components):
    matrix = interaction_matrix()
    model = NMF(
        n_components=n_components,
        init="nndsvda",
        random_state=123,
        max_iter=500,
    )
    user_features = model.fit_transform(matrix.to_numpy(dtype=float))
    item_features = model.components_
    return model, matrix, user_features, item_features


def course_similarity_scores(selected_ids, threshold=0.0):
    id_to_idx, sim_matrix = prepare_similarity_map()
    scores = {}

    for selected_id in selected_ids:
        if selected_id not in id_to_idx:
            continue

        selected_idx = id_to_idx[selected_id]

        for candidate_id, candidate_idx in id_to_idx.items():
            if candidate_id in selected_ids:
                continue

            score = float(sim_matrix[selected_idx, candidate_idx])
            if score >= threshold:
                scores[candidate_id] = max(score, scores.get(candidate_id, -np.inf))

    return scores


def user_profile_scores(selected_ids, threshold=0.0):
    learner_vector, genre_columns = selected_course_genre_profile(selected_ids)
    candidates = genres[~genres["COURSE_ID"].isin(selected_ids)].copy()
    raw_scores = candidates[genre_columns].to_numpy(dtype=float) @ learner_vector

    return {
        course_id: float(score)
        for course_id, score in zip(candidates["COURSE_ID"], raw_scores)
        if score >= threshold
    }


def hybrid_scores(selected_ids, similarity_weight):
    """
    App-level blend of two completed content-based methods.
    This is not presented as a separate lab model.
    """
    similarity = course_similarity_scores(selected_ids, threshold=0.0)
    profile = user_profile_scores(selected_ids, threshold=0.0)

    sim_norm = minmax_score_map(similarity)
    profile_norm = minmax_score_map(profile)

    candidate_ids = set(sim_norm) | set(profile_norm)
    combined = {}
    reasons = {}
    support = {}

    for course_id in candidate_ids:
        sim_value = sim_norm.get(course_id, 0.0)
        profile_value = profile_norm.get(course_id, 0.0)
        combined[course_id] = (
            similarity_weight * sim_value
            + (1 - similarity_weight) * profile_value
        )

        supported_by = []
        if sim_value > 0:
            supported_by.append("course similarity")
        if profile_value > 0:
            supported_by.append("learner interests")

        if len(supported_by) == 2:
            reasons[course_id] = (
                "Supported by both course-content similarity and the interests "
                "found in your selected courses."
            )
            support[course_id] = "Evidence from 2 methods."
        elif supported_by:
            reasons[course_id] = f"Supported mainly by {supported_by[0]}."
            support[course_id] = "Evidence from 1 method."
        else:
            reasons[course_id] = "Ranked by the combined content-based view."
            support[course_id] = "Limited supporting evidence."

    return combined, reasons, support


def clustering_scores(selected_ids, use_pca, n_clusters, minimum_enrollments):
    learner_vector, genre_columns = selected_course_genre_profile(selected_ids)
    model, reducer, feature_columns, popularity, component_count = train_clusters(
        use_pca,
        n_clusters,
    )

    aligned = pd.Series(0.0, index=feature_columns)
    for name, value in zip(genre_columns, learner_vector):
        if name in aligned.index:
            aligned.loc[name] = value

    learner_x = aligned.to_numpy(dtype=float).reshape(1, -1)
    learner_model_x = reducer.transform(learner_x) if reducer is not None else learner_x
    cluster = int(model.predict(learner_model_x)[0])

    candidates = popularity[
        (popularity["cluster"] == cluster)
        & (popularity["enrollments"] >= minimum_enrollments)
        & (~popularity["item"].isin(selected_ids))
    ]

    scores = dict(zip(candidates["item"], candidates["enrollments"]))
    return scores, cluster, component_count


def knn_scores(selected_ids, k, minimum_similarity):
    matrix = interaction_matrix()
    available = [course_id for course_id in selected_ids if course_id in matrix.columns]

    if not available:
        return {}

    item_matrix = matrix.T
    learner_vector = (
        item_matrix.loc[available]
        .mean(axis=0)
        .to_numpy()
        .reshape(1, -1)
    )

    neighbours = min(k + len(available), len(item_matrix))
    model = NearestNeighbors(
        n_neighbors=neighbours,
        metric="cosine",
        algorithm="brute",
    )
    model.fit(item_matrix.to_numpy(dtype=float))
    distances, indices = model.kneighbors(learner_vector)

    scores = {}
    for distance, index in zip(distances[0], indices[0]):
        course_id = item_matrix.index[index]
        similarity = 1 - float(distance)

        if (
            course_id not in selected_ids
            and similarity >= minimum_similarity
        ):
            scores[course_id] = similarity

    return scores


def nmf_scores(selected_ids, n_components, minimum_score):
    model, matrix, _, item_features = train_nmf(n_components)
    learner_row = np.zeros((1, matrix.shape[1]), dtype=float)

    for course_id in selected_ids:
        if course_id in matrix.columns:
            learner_row[0, matrix.columns.get_loc(course_id)] = 3.0

    learner_features = model.transform(learner_row)
    predictions = learner_features @ item_features

    return {
        course_id: float(score)
        for course_id, score in zip(matrix.columns, predictions.ravel())
        if course_id not in selected_ids and score >= minimum_score
    }


def detect_id_column(df, preferred):
    for column in preferred:
        if column in df.columns:
            return column
    return df.columns[0]


def embedding_feature_columns(df, id_column):
    return [
        column
        for column in df.columns
        if column != id_column and pd.api.types.is_numeric_dtype(df[column])
    ]


def neural_embedding_scores(existing_user_id, selected_ids, minimum_score):
    user_id_column = detect_id_column(
        user_embeddings,
        ["user", "USER", "USER_ID"],
    )
    course_id_column = detect_id_column(
        course_embeddings,
        ["item", "COURSE_ID", "ITEM", "ITEM_ID"],
    )

    user_columns = embedding_feature_columns(user_embeddings, user_id_column)
    course_columns = embedding_feature_columns(
        course_embeddings,
        course_id_column,
    )
    vector_size = min(len(user_columns), len(course_columns))

    user_row = user_embeddings[
        user_embeddings[user_id_column] == existing_user_id
    ]
    if user_row.empty:
        return {}

    user_vector = user_row.iloc[0][
        user_columns[:vector_size]
    ].to_numpy(dtype=float)
    course_matrix = course_embeddings[
        course_columns[:vector_size]
    ].to_numpy(dtype=float)

    scores_array = course_matrix @ user_vector

    return {
        course_id: float(score)
        for course_id, score in zip(
            course_embeddings[course_id_column],
            scores_array,
        )
        if course_id not in selected_ids and score >= minimum_score
    }


@st.cache_resource(show_spinner="Evaluating the embedding regression model...")
def evaluate_embedding_regression(model_name, alpha, l1_ratio):
    user_id_column = detect_id_column(
        user_embeddings,
        ["user", "USER", "USER_ID"],
    )
    course_id_column = detect_id_column(
        course_embeddings,
        ["item", "COURSE_ID", "ITEM", "ITEM_ID"],
    )

    merged = ratings.merge(
        user_embeddings,
        left_on="user",
        right_on=user_id_column,
        how="inner",
    ).merge(
        course_embeddings,
        left_on="item",
        right_on=course_id_column,
        how="inner",
        suffixes=("_user", "_course"),
    )

    excluded = {"user", "item", "rating", user_id_column, course_id_column}
    feature_columns = [
        column
        for column in merged.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(merged[column])
    ]

    x = merged[feature_columns].to_numpy(dtype=float)
    y = merged["rating"].to_numpy(dtype=float)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=alpha),
        "Lasso": Lasso(alpha=alpha, max_iter=10000),
        "ElasticNet": ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=10000,
        ),
    }

    model = models[model_name]
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    return rmse


@st.cache_resource(show_spinner="Evaluating the embedding classification model...")
def evaluate_embedding_classification(c_value, class_weight):
    user_id_column = detect_id_column(
        user_embeddings,
        ["user", "USER", "USER_ID"],
    )
    course_id_column = detect_id_column(
        course_embeddings,
        ["item", "COURSE_ID", "ITEM", "ITEM_ID"],
    )

    merged = ratings.merge(
        user_embeddings,
        left_on="user",
        right_on=user_id_column,
        how="inner",
    ).merge(
        course_embeddings,
        left_on="item",
        right_on=course_id_column,
        how="inner",
        suffixes=("_user", "_course"),
    )

    excluded = {"user", "item", "rating", user_id_column, course_id_column}
    feature_columns = [
        column
        for column in merged.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(merged[column])
    ]

    x = merged[feature_columns].to_numpy(dtype=float)
    encoder = LabelEncoder()
    y = encoder.fit_transform(merged["rating"])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(
        C=c_value,
        class_weight=None if class_weight == "None" else "balanced",
        solver="lbfgs",
        max_iter=2000,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1 score": float(f1),
    }


def relaxed_retry(method, selected_ids, settings):
    """A transparent fallback when the user's chosen threshold is too strict."""
    if method == "Course Similarity":
        return course_similarity_scores(selected_ids, threshold=0.0)
    if method == "User Profile":
        return user_profile_scores(selected_ids, threshold=0.0)
    if method == "KNN":
        return knn_scores(
            selected_ids,
            max(settings.get("k", 20), 40),
            minimum_similarity=0.0,
        )
    if method == "NMF":
        return nmf_scores(
            selected_ids,
            settings.get("components", 16),
            minimum_score=0.0,
        )
    return {}


# =========================================================
# Main tabs
# =========================================================
recommend_tab, evidence_tab, guide_tab = st.tabs(
    [
        "🎯 Get recommendations",
        "📊 Evidence & limitations",
        "🧭 Beginner guide",
    ]
)

# =========================================================
# Recommendations
# =========================================================
with recommend_tab:
    st.markdown(
        """
        <div class="guide">
          <strong>Best starting point:</strong>
          select 3–6 courses that genuinely represent your interests, keep
          <em>Hybrid view</em> selected, and request 10 recommendations.
          One selected course can work, but it provides less evidence about
          your overall interests.
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_mode = st.radio(
        "Step 1 — How should the app understand the learner?",
        ["Select completed courses", "Use an existing learner ID"],
        horizontal=True,
    )

    sorted_courses = course_lookup.sort_values("TITLE")
    label_to_id = {
        f"{row.TITLE} · {row.COURSE_ID}": row.COURSE_ID
        for row in sorted_courses.itertuples()
    }

    selected_ids = []
    existing_user = None

    if source_mode == "Select completed courses":
        chosen_labels = st.multiselect(
            "Choose courses you have completed or audited",
            options=list(label_to_id),
            placeholder="Search by course title",
        )
        selected_ids = [label_to_id[label] for label in chosen_labels]
    else:
        embedding_user_column = detect_id_column(
            user_embeddings,
            ["user", "USER", "USER_ID"],
        )
        available_users = sorted(
            set(ratings["user"]).intersection(
                set(user_embeddings[embedding_user_column])
            )
        )
        existing_user = st.selectbox("Choose an existing learner ID", available_users)
        selected_ids = (
            ratings.loc[ratings["user"] == existing_user, "item"]
            .drop_duplicates()
            .tolist()
        )

    if 0 < len(selected_ids) < 3:
        st.warning(
            "The app can continue, but one or two courses give limited evidence. "
            "Selecting at least three related courses usually creates a more stable profile."
        )

    if selected_ids:
        selected_display = course_lookup[
            course_lookup["COURSE_ID"].isin(selected_ids)
        ][["COURSE_ID", "TITLE"]]

        with st.expander(
            f"Selected learning history ({len(selected_display)} courses)",
            expanded=False,
        ):
            st.dataframe(
                selected_display,
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.info("Select at least one course to continue.")

    methods = [
        "Hybrid view (recommended)",
        "Course Similarity",
        "User Profile",
        "Clustering",
        "Clustering with PCA",
        "KNN",
        "NMF",
        "Neural Network Embeddings",
        "Regression with Embedding Features",
        "Classification with Embedding Features",
    ]

    first, second, third = st.columns([1.15, 1, 1])

    with first:
        method = st.selectbox("Step 2 — Choose a method", methods)
        top_n = st.slider("Number of results", 5, 25, 10)

    settings = {}

    with second:
        if method == "Hybrid view (recommended)":
            settings["similarity_weight"] = st.slider(
                "Weight given to course similarity",
                min_value=0.0,
                max_value=1.0,
                value=0.60,
                step=0.05,
                help=(
                    "The remaining weight is given to the user-profile method. "
                    "This combines two completed content-based approaches."
                ),
            )
        elif method == "Course Similarity":
            settings["threshold"] = st.slider(
                "Minimum raw similarity",
                0.0,
                1.0,
                0.30,
                0.05,
            )
        elif method == "User Profile":
            settings["threshold"] = st.slider(
                "Minimum profile score",
                0.0,
                30.0,
                3.0,
                0.5,
            )
        elif method in ["Clustering", "Clustering with PCA"]:
            settings["clusters"] = st.slider("Number of clusters", 2, 15, 6)
            settings["minimum_enrollments"] = st.slider(
                "Minimum enrollments in the matched cluster",
                0,
                30,
                5,
            )
        elif method == "KNN":
            settings["k"] = st.slider("Number of neighbours", 5, 60, 40)
            settings["minimum_similarity"] = st.slider(
                "Minimum neighbour similarity",
                0.0,
                1.0,
                0.0,
                0.05,
            )
        elif method == "NMF":
            settings["components"] = st.slider(
                "Latent features",
                4,
                32,
                16,
                4,
            )
            settings["minimum_score"] = st.slider(
                "Minimum predicted score",
                0.0,
                3.0,
                0.0,
                0.10,
            )
        elif method == "Neural Network Embeddings":
            settings["minimum_score"] = st.slider(
                "Minimum embedding score",
                -2.0,
                3.0,
                -2.0,
                0.1,
            )
        elif method == "Regression with Embedding Features":
            settings["regression_model"] = st.selectbox(
                "Regression model",
                ["Linear Regression", "Ridge", "Lasso", "ElasticNet"],
            )
            settings["alpha"] = st.select_slider(
                "Alpha",
                [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                value=0.1,
            )
            settings["l1_ratio"] = st.slider(
                "ElasticNet L1 ratio",
                0.1,
                0.9,
                0.5,
                0.1,
            )
        else:
            settings["c"] = st.select_slider(
                "Logistic Regression C",
                [0.01, 0.1, 1.0, 10.0, 100.0],
                value=1.0,
            )
            settings["class_weight"] = st.selectbox(
                "Class weight",
                ["None", "balanced"],
            )

    with third:
        st.markdown("**What the result means**")
        method_help = {
            "Hybrid view (recommended)": (
                "Combines normalized rankings from Course Similarity and User Profile. "
                "This is an app-level comparison view, not a new lab model."
            ),
            "Course Similarity": (
                "Ranks unseen courses by how similar their course content is "
                "to the courses you selected."
            ),
            "User Profile": (
                "Builds an interest profile from the genres of your selected courses."
            ),
            "Clustering": (
                "Matches the learner to a group and recommends popular unseen courses "
                "inside that group."
            ),
            "Clustering with PCA": (
                "Reduces the user-profile features before matching the learner to a group."
            ),
            "KNN": (
                "Uses similarities in user-course interaction patterns."
            ),
            "NMF": (
                "Uses smaller latent representations learned from the user-item matrix."
            ),
            "Neural Network Embeddings": (
                "Uses learned user and course vectors. It requires an existing learner ID."
            ),
            "Regression with Embedding Features": (
                "Evaluates numerical rating prediction. It does not create a new-user "
                "ranking when the required embedding rows are unavailable."
            ),
            "Classification with Embedding Features": (
                "Evaluates rating-category prediction rather than direct course ranking."
            ),
        }
        st.caption(method_help[method])

    existing_user_required = method in [
        "Neural Network Embeddings",
        "Regression with Embedding Features",
        "Classification with Embedding Features",
    ]

    restricted = existing_user_required and source_mode != "Use an existing learner ID"

    if restricted:
        st.warning(
            "This method requires an existing learner embedding. Switch to "
            "**Use an existing learner ID**. The app will not invent an embedding "
            "for a new learner."
        )

    generate = st.button(
        "Step 3 — Generate recommendations",
        type="primary",
        use_container_width=True,
        disabled=(not selected_ids or restricted),
    )

    if generate:
        raw_scores = {}
        reason_map = None
        support_map = None
        note = ""
        score_method_name = method

        with st.spinner(f"Running {method}..."):
            if method == "Hybrid view (recommended)":
                raw_scores, reason_map, support_map = hybrid_scores(
                    selected_ids,
                    settings["similarity_weight"],
                )

            elif method == "Course Similarity":
                raw_scores = course_similarity_scores(
                    selected_ids,
                    settings["threshold"],
                )

            elif method == "User Profile":
                raw_scores = user_profile_scores(
                    selected_ids,
                    settings["threshold"],
                )

            elif method in ["Clustering", "Clustering with PCA"]:
                raw_scores, cluster, component_count = clustering_scores(
                    selected_ids,
                    use_pca=(method == "Clustering with PCA"),
                    n_clusters=settings["clusters"],
                    minimum_enrollments=settings["minimum_enrollments"],
                )
                note = f"The learner was matched to cluster {cluster}."
                if method == "Clustering with PCA":
                    note += (
                        f" PCA retained {component_count} components to reach "
                        "at least 90% cumulative explained variance."
                    )

            elif method == "KNN":
                raw_scores = knn_scores(
                    selected_ids,
                    settings["k"],
                    settings["minimum_similarity"],
                )

            elif method == "NMF":
                raw_scores = nmf_scores(
                    selected_ids,
                    settings["components"],
                    settings["minimum_score"],
                )

            elif method == "Neural Network Embeddings":
                raw_scores = neural_embedding_scores(
                    existing_user,
                    selected_ids,
                    settings["minimum_score"],
                )

            elif method == "Regression with Embedding Features":
                rmse = evaluate_embedding_regression(
                    settings["regression_model"],
                    settings["alpha"],
                    settings["l1_ratio"],
                )
                note = (
                    f"The holdout RMSE is {rmse:.4f}. This evaluates rating "
                    "prediction error; it is not a confidence score for one "
                    "downloaded recommendation list."
                )

            else:
                metrics = evaluate_embedding_classification(
                    settings["c"],
                    settings["class_weight"],
                )
                note = (
                    "The classification model was evaluated on a holdout split. "
                    "See the Evidence tab for accuracy, precision, recall and F1."
                )

        if note:
            st.info(note)

        if method in [
            "Regression with Embedding Features",
            "Classification with Embedding Features",
        ]:
            st.info(
                "This option evaluates its model honestly but does not invent "
                "course-level recommendations when the required new learner-course "
                "embedding pairs are unavailable."
            )
        else:
            used_relaxed_settings = False

            if not raw_scores and method in [
                "Course Similarity",
                "User Profile",
                "KNN",
                "NMF",
            ]:
                raw_scores = relaxed_retry(method, selected_ids, settings)
                used_relaxed_settings = bool(raw_scores)

            if used_relaxed_settings:
                st.warning(
                    "Nothing passed your original threshold, so the app retried "
                    "with a relaxed threshold. The displayed results are clearly "
                    "ranked, but you should review them more carefully."
                )

            results = build_results(
                raw_scores,
                selected_ids,
                top_n,
                method=score_method_name,
                reason_map=reason_map,
                secondary_map=support_map,
            )

            if results.empty:
                st.warning(
                    "No suitable candidates were found. Select more completed "
                    "courses or try the Hybrid view, Course Similarity or User Profile."
                )
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Recommendations", len(results))
                m2.metric(
                    "Top match strength",
                    f"{results['MATCH_STRENGTH'].max():.1f}/100",
                )
                m3.metric(
                    "Evidence base",
                    f"{len(selected_ids)} selected courses",
                )
                m4.metric("Method", method)

                st.caption(
                    "**Match strength is a relative 0–100 display score within this "
                    "result list. It is not accuracy, probability or RMSE.** "
                    "Raw scores remain visible so the model output is not hidden."
                )

                display_columns = [
                    "RANK",
                    "TITLE",
                    "COURSE_ID",
                    "MATCH_STRENGTH",
                    "CONFIDENCE",
                    "WHY_RECOMMENDED",
                    "RAW_SCORE",
                ]

                st.dataframe(
                    results[display_columns],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "MATCH_STRENGTH": st.column_config.ProgressColumn(
                            "Relative match strength",
                            min_value=0.0,
                            max_value=100.0,
                            format="%.1f",
                        ),
                        "RAW_SCORE": st.column_config.NumberColumn(
                            "Raw model score",
                            format="%.4f",
                        ),
                    },
                )

                chart_df = results.copy()
                chart_df["LABEL"] = chart_df["TITLE"].fillna(
                    chart_df["COURSE_ID"]
                )
                st.subheader("Recommendation ranking")
                st.bar_chart(
                    chart_df.set_index("LABEL")["MATCH_STRENGTH"],
                    horizontal=True,
                )

                with st.expander("How should I judge these results?", expanded=True):
                    st.markdown(
                        """
                        - Start with the **rank and explanation**, not the raw decimal alone.
                        - A stronger result is supported by several selected courses or,
                          in the Hybrid view, by both content-based methods.
                        - Read the title and description before enrolling. The model
                          does not know your schedule, budget, prior knowledge or career deadline.
                        - Low raw scores can still produce a useful ranking when every
                          available course has a similar score.
                        - Final quality can only be confirmed after the learner reviews,
                          takes or rates the recommended course.
                        """
                    )

                st.download_button(
                    "Download recommendation list",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name=(
                        method.lower()
                        .replace(" ", "_")
                        .replace("(", "")
                        .replace(")", "")
                        + "_recommendations.csv"
                    ),
                    mime="text/csv",
                )

# =========================================================
# Evidence and limitations
# =========================================================
with evidence_tab:
    st.subheader("What evidence is available?")

    st.markdown(
        """
        <div class="guide">
          <strong>Important distinction:</strong>
          model evaluation measures how a method performed on historical
          test data. It does not prove that every future recommendation will
          suit every learner.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="soft-card">
              <h4>Raw recommendation score</h4>
              <p>Produced by the selected method and used to rank courses. Its scale changes across methods.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="soft-card">
              <h4>Relative match strength</h4>
              <p>A 0–100 view created only to compare courses inside the current result list. It is not accuracy.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="soft-card">
              <h4>Model evaluation</h4>
              <p>RMSE evaluates numerical rating predictions; accuracy, precision, recall and F1 evaluate rating categories.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    evidence_type = st.selectbox(
        "Choose an evidence check",
        [
            "Regression with embedding features",
            "Classification with embedding features",
            "Review a recommendation CSV",
        ],
    )

    if evidence_type == "Regression with embedding features":
        model_name = st.selectbox(
            "Regression model",
            ["Linear Regression", "Ridge", "Lasso", "ElasticNet"],
            key="evidence_regression_model",
        )
        alpha = st.select_slider(
            "Alpha",
            [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            value=0.1,
            key="evidence_alpha",
        )
        l1_ratio = st.slider(
            "L1 ratio",
            0.1,
            0.9,
            0.5,
            0.1,
            key="evidence_l1_ratio",
        )

        if st.button("Evaluate regression", key="run_regression_evidence"):
            rmse = evaluate_embedding_regression(
                model_name,
                alpha,
                l1_ratio,
            )
            st.metric("Holdout RMSE", f"{rmse:.4f}")
            st.caption(
                "Lower RMSE means the predicted ratings were closer to the "
                "historical test ratings. It does not measure the quality of a "
                "single recommendation CSV that lacks actual ratings."
            )

    elif evidence_type == "Classification with embedding features":
        c_value = st.select_slider(
            "C",
            [0.01, 0.1, 1.0, 10.0, 100.0],
            value=1.0,
            key="evidence_c",
        )
        class_weight = st.selectbox(
            "Class weight",
            ["None", "balanced"],
            key="evidence_class_weight",
        )

        if st.button("Evaluate classification", key="run_classification_evidence"):
            metrics = evaluate_embedding_classification(
                c_value,
                class_weight,
            )
            columns = st.columns(4)

            for column, (metric_name, metric_value) in zip(
                columns,
                metrics.items(),
            ):
                column.metric(metric_name, f"{metric_value:.4f}")

            metric_frame = pd.DataFrame(
                {
                    "Metric": list(metrics),
                    "Score": list(metrics.values()),
                }
            )
            st.bar_chart(metric_frame.set_index("Metric"))

    else:
        uploaded = st.file_uploader(
            "Upload a generated recommendation CSV",
            type=["csv"],
        )

        if uploaded:
            result_df = pd.read_csv(uploaded)
            st.dataframe(
                result_df,
                hide_index=True,
                use_container_width=True,
            )

            uppercase = {column.upper(): column for column in result_df.columns}
            score_column = None

            for candidate in [
                "MATCH_STRENGTH",
                "SCORE",
                "RAW_SCORE",
                "PREDICTED_RATING",
            ]:
                if candidate in uppercase:
                    score_column = uppercase[candidate]
                    break

            user_column = uppercase.get("USER")
            course_column = uppercase.get("COURSE_ID")

            if user_column and course_column:
                average_count = (
                    result_df.groupby(user_column)[course_column]
                    .nunique()
                    .mean()
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("Learners", result_df[user_column].nunique())
                m2.metric(
                    "Unique recommended courses",
                    result_df[course_column].nunique(),
                )
                m3.metric(
                    "Average recommendations per learner",
                    f"{average_count:.2f}",
                )

                top_courses = (
                    result_df.groupby(course_column)[user_column]
                    .nunique()
                    .sort_values(ascending=False)
                    .head(10)
                )
                st.subheader("Most frequently recommended courses")
                st.bar_chart(top_courses, horizontal=True)

            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Rows", len(result_df))
                m2.metric(
                    "Courses",
                    (
                        result_df[course_column].nunique()
                        if course_column
                        else len(result_df)
                    ),
                )

                if score_column and pd.api.types.is_numeric_dtype(
                    result_df[score_column]
                ):
                    m3.metric(
                        "Average displayed score",
                        f"{result_df[score_column].mean():.3f}",
                    )
                    st.subheader("Score distribution")
                    st.bar_chart(
                        result_df[[score_column]]
                        .reset_index(drop=True)
                    )
                else:
                    m3.metric("Score column", "Not found")

                st.info(
                    "This appears to be a single-learner recommendation file. "
                    "The app can summarize its ranking, but it cannot calculate "
                    "RMSE unless the file contains both actual and predicted ratings."
                )

            actual_column = uppercase.get("ACTUAL_RATING")
            predicted_column = uppercase.get("PREDICTED_RATING")

            if actual_column and predicted_column:
                upload_rmse = float(
                    np.sqrt(
                        mean_squared_error(
                            result_df[actual_column],
                            result_df[predicted_column],
                        )
                    )
                )
                st.metric("RMSE from uploaded actual/predicted ratings", f"{upload_rmse:.4f}")

    st.divider()
    st.subheader("Honest strengths and limitations")

    limitations = pd.DataFrame(
        [
            [
                "Course Similarity",
                "Easy to explain and works from selected course content.",
                "May become narrow and recommend courses too similar to what the learner already knows.",
            ],
            [
                "User Profile",
                "Uses several selected courses to summarize interests.",
                "Quality depends on how well selected courses represent the learner.",
            ],
            [
                "Clustering",
                "Uses patterns from groups of similar learners.",
                "A cluster is a broad group and does not capture every personal preference.",
            ],
            [
                "KNN",
                "Finds nearby interaction patterns without a complex model.",
                "Can return weak or no results when interaction overlap is limited.",
            ],
            [
                "NMF",
                "Finds compact hidden patterns in the user-item matrix.",
                "Latent features are difficult to explain directly to users.",
            ],
            [
                "Embeddings",
                "Represent users and courses with learned compact vectors.",
                "Existing embeddings are needed; a new user has a cold-start problem.",
            ],
        ],
        columns=["Method", "Strength", "Limitation"],
    )
    st.dataframe(limitations, hide_index=True, use_container_width=True)

# =========================================================
# Beginner guide
# =========================================================
with guide_tab:
    st.subheader("How to use the app")

    usage_steps = [
        (
            "1. Select your real learning history",
            "Choose 3–6 completed courses that genuinely represent the subjects you want to continue learning.",
        ),
        (
            "2. Start with the Hybrid view",
            "It combines the Course Similarity and User Profile rankings. The app clearly labels this as a combined view rather than a separate trained lab model.",
        ),
        (
            "3. Keep the default settings first",
            "Generate a baseline result before changing thresholds, neighbours, clusters or latent features.",
        ),
        (
            "4. Read the explanation column",
            "The reason is more useful than treating a raw decimal as a universal quality score.",
        ),
        (
            "5. Compare another method",
            "Try Course Similarity, User Profile, KNN or NMF and check whether important courses appear repeatedly.",
        ),
        (
            "6. Review before enrolling",
            "The app cannot know your available time, cost, prior knowledge or personal career constraints.",
        ),
    ]

    for title, description in usage_steps:
        st.markdown(
            f'<div class="step"><strong>{title}</strong><br>{description}</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("How the prediction is built")

    prediction_steps = [
        [
            "Learner input",
            "Completed courses or an existing learner ID",
        ],
        [
            "Numerical representation",
            "Course genres, Bag of Words, interaction patterns, clusters or embeddings",
        ],
        [
            "Recommendation calculation",
            "Similarity, profile matching, neighbours, clustering, matrix factorization or embedding interaction",
        ],
        [
            "Filtering",
            "Courses already completed are removed",
        ],
        [
            "Ranking",
            "Remaining courses are sorted using the method-specific raw score",
        ],
        [
            "User display",
            "Rank, raw score, relative match strength and a short explanation",
        ],
    ]

    st.dataframe(
        pd.DataFrame(
            prediction_steps,
            columns=["Stage", "What happens"],
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "A recommendation is a decision aid, not a guarantee. The best final "
        "check is whether the course description, difficulty and learning goal "
        "fit the learner's real needs."
    )

st.caption(
    "Independent portfolio project · Recommendations are ranked from historical "
    "course data and should be reviewed by the learner."
)
