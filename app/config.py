"""Configuration and artifact paths for the local demo and Hugging Face Space."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = Path(__file__).resolve().parent

SPACE_NAME = "nepali-news-demo"
MODEL_REPO_ID = "Nisxal/nepali-nlp-xlmr-matched50k-a"
REPORT_URL = "<report-link>"
GITHUB_URL = "<github-repository-link>"

MINILM_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
XLMR_MODEL_DIR = APP_ROOT / "model_repo" / "xlmr-matched50k-A"

TFIDF_BUNDLE = (
    APP_ROOT / "reference_calibrated_bundle.joblib"
    if (APP_ROOT / "reference_calibrated_bundle.joblib").exists()
    else ROOT / "results" / "models" / "reference_calibrated_bundle.joblib"
)
MINILM_BUNDLE = (
    APP_ROOT / "sentence_embedding_reference_bundle.joblib"
    if (APP_ROOT / "sentence_embedding_reference_bundle.joblib").exists()
    else ROOT / "results" / "models" / "sentence_embedding_reference_bundle.joblib"
)
RESEARCH_DATA = APP_ROOT / "research_data"
HELDOUT_TABLE = RESEARCH_DATA / "source_held_out_summary.csv"
MATCHED_TABLE = RESEARCH_DATA / "matched_50k_ab_comparison.csv"
FINAL_METRICS = RESEARCH_DATA / "final_test_metrics.json"
FIGURE_PATH = RESEARCH_DATA / "paper_heldout_publisher_macro_f1.png"
XLMR_HELDOUT_STATUSES = sorted(RESEARCH_DATA.glob("matched50k_xlmr_C_*_status.json"))

LABELS = (
    "Arts", "Automobile", "Bank", "Business", "Crime", "Economy",
    "Education", "Employment", "Entertainment", "Health", "Lifestyle",
    "Literature", "Market", "Opinion", "Politics", "Society", "Sports",
    "Technology", "Tourism", "World",
)

EXAMPLES = (
    "नेपालको महिला फुटबल टोलीले अन्तर्राष्ट्रिय प्रतियोगिताका लागि तयारी गरिरहेको छ।",
    "नेपाल राष्ट्र बैंकले मौद्रिक नीतिमा नयाँ व्यवस्था सार्वजनिक गरेको छ।",
    "सरकारले विद्यालय शिक्षाको गुणस्तर सुधार गर्न नयाँ कार्यक्रम घोषणा गरेको छ।",
    "सहरमा नयाँ विद्युतीय सवारी साधनको बिक्री बढ्दै गएको व्यवसायीहरूले बताएका छन्।",
    "स्वास्थ्य मन्त्रालयले डेंगु नियन्त्रणका लागि सावधानी अपनाउन आग्रह गरेको छ।",
    "पर्यटन व्यवसायीले हिमाली क्षेत्रमा पदयात्रा मौसम राम्रो रहेको बताएका छन्।",
)

DISCLAIMER = (
    "यो अनुसन्धान प्रदर्शन मात्र हो। वर्गीकरण गलत हुन सक्छ र यसलाई तथ्य-जाँच, "
    "समाचार सम्पादन, कानुनी, चिकित्सकीय वा अन्य व्यावसायिक सल्लाहका रूपमा प्रयोग नगर्नुहोस्। "
    "निजी वा संवेदनशील पाठ पेश नगर्नुहोस्।"
)
