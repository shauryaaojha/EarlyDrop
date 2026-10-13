"""Project-wide constants: paths, seed, form schema and the modelling feature set."""
from pathlib import Path

SEED = 42
N_STUDENTS = 400
DATA_IS_SYNTHETIC = True  # the modelling cohort is synthetic; shown in dashboard + deck

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
REAL = DATA / "real"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
FIGS = ROOT / "figs"

REAL_FORM_CSV = REAL / "form_responses_real.csv"
SYN_FORM_CSV = RAW / "form_responses_synthetic.csv"
SYN_OUTCOMES_CSV = RAW / "outcomes_synthetic.csv"
SYN_HISTORY_CSV = RAW / "semester_history_synthetic.csv"
GROUND_TRUTH_JSON = RAW / "ground_truth.json"
CLEAN_CSV = PROCESSED / "earlydrop_clean.csv"
REAL_CLEAN_CSV = PROCESSED / "real_clean.csv"
METRICS_JSON = RESULTS / "metrics.json"

# Google Form headers, in form order, mapped to snake_case names.
# Taken verbatim from the live EarlyDrop form export (50 columns).
FORM_COLUMNS = [
    ("Timestamp", "timestamp"),
    ("Programme/Degree", "programme"),
    ("Department", "department"),
    ("Current year", "current_year"),
    ("Current semester", "current_semester"),
    ("Previous semester SGPA (0.0 - 10.0)", "previous_sem_sgpa"),
    ("Current CGPA (0.0 - 10.0)", "current_cgpa"),
    ("Average internal assessment score (Percentage)", "internal_assessment_average"),
    ("Overall attendance percentage", "attendance_percentage"),
    ("Number of courses successfully completed", "courses_completed"),
    ("Current active backlogs/arrears", "active_backlogs"),
    ("Academic performance compared with previous semester", "academic_performance_change"),
    ("Assignment completion consistency (1 = Inconsistent, 5 = Highly consistent)", "assignment_completion_consistency"),
    ("Regularity of class attendance (1 = Rarely, 5 = Always)", "class_attendance_frequency"),
    ("Recent attendance change", "attendance_change"),
    ("Class participation (1 = Very low, 5 = Very high)", "class_participation"),
    ("Frequency of asking or clarifying doubts (1 = Never, 5 = Very frequently)", "doubt_resolution_frequency"),
    ("Frequency of faculty interaction regarding academics (1 = Never, 5 = Very frequently)", "faculty_academic_interaction"),
    ("Study hours outside classes per day", "daily_study_hours"),
    ("Consistency of study routine (1 = Not consistent, 5 = Very consistent)", "study_consistency"),
    ("Studying mainly just before examinations (1 = Never, 5 = Almost always)", "exam_cramming_frequency"),
    ("Studying beyond material directly covered in class (1 = Never, 5 = Very frequently)", "self_learning_frequency"),
    ("Frequency of revising previously taught topics (1 = Never, 5 = Very frequently)", "revision_frequency"),
    ("Use of additional academic resources (1 = Never, 5 = Very frequently)", "additional_resource_usage"),
    ("Frequency of accessing college LMS/academic portal (1 = Never, 5 = Very frequently)", "lms_usage_frequency"),
    ("Frequency of accessing online learning materials (1 = Never, 5 = Very frequently)", "online_learning_material_usage"),
    ("Consistency of completing online academic activities (1 = Inconsistent, 5 = Highly consistent)", "online_activity_completion"),
    ("Online academic learning hours per week", "weekly_online_learning_hours"),
    ("Recent digital academic engagement", "digital_engagement_change"),
    ("Frequency of late assignment submission (1 = Never, 5 = Very frequently)", "late_assignment_frequency"),
    ("Frequency of missing quizzes/tests/assessments", "missed_assessment_frequency"),
    ("Regularity of test preparation (1 = Not regular, 5 = Very regular)", "test_preparation_consistency"),
    ("Recent assessment performance", "assessment_performance_trend"),
    ("Interest in current academic subjects (1 = Very low, 5 = Very high)", "course_interest"),
    ("Motivation toward academic work (1 = Very low, 5 = Very high)", "academic_motivation"),
    ("Clarity of academic/career goals (1 = No clarity, 5 = Very clear)", "career_goal_clarity"),
    ("Satisfaction with current academic progress (1 = Very dissatisfied, 5 = Very satisfied)", "academic_progress_satisfaction"),
    ("Overall engagement with studies (1 = Very low, 5 = Very high)", "overall_academic_engagement"),
    ("Confidence in successfully completing the programme (1 = Not confident, 5 = Highly confident)", "programme_completion_confidence"),
    ("Academic support from peers (1 = Very poor, 5 = Excellent)", "peer_support"),
    ("Academic support from faculty (1 = Very poor, 5 = Excellent)", "faculty_support"),
    ("Ease of seeking academic help (1 = Very difficult, 5 = Very easy)", "academic_help_accessibility"),
    ("Frequency of receiving academic guidance when facing difficulties (1 = Never, 5 = Very frequently)", "academic_guidance_frequency"),
    ("Difficulty of current academic workload (1 = Very easy, 5 = Very difficult)", "academic_workload_difficulty"),
    ("Concern about current academic performance (1 = Not concerned, 5 = Extremely concerned)", "academic_performance_concern"),
    ("Recent overall academic situation", "overall_academic_situation_change"),
    ("Need for additional academic support", "additional_academic_support_needed"),
    # The live form repeats the confidence question; the second copy is treated as "certainty".
    ("Confidence in successfully completing the programme (1 = Not confident, 5 = Highly confident)", "programme_completion_certainty"),
    ("Biggest academic challenge currently faced", "biggest_academic_challenge"),
    ("Type of academic support that would be most useful", "required_academic_support"),
]
HEADERS = [h for h, _ in FORM_COLUMNS]
SNAKE = [s for _, s in FORM_COLUMNS]

# Questions added to the form after the first responses came in (blank for early rows).
LATE_ADDED = [
    "internal_assessment_average", "attendance_change", "exam_cramming_frequency",
    "additional_resource_usage", "lms_usage_frequency", "digital_engagement_change",
    "academic_motivation", "career_goal_clarity", "academic_progress_satisfaction",
]

# Text answer -> number encodings (answer wording copied from the live form).
ORDINAL_MAPS = {
    "current_year": {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4},
    "active_backlogs": {"0": 0, "1": 1, "2": 2, "3": 3, "4 or more": 4},
    "academic_performance_change": {"Significantly declined": 1, "Declined": 2, "Same": 3,
                                    "Improved": 4, "Significantly improved": 5},
    "attendance_change": {"Decreased": 1, "Same": 2, "Increased": 3},
    "daily_study_hours": {"<1": 0.5, "1-2": 1.5, "2-3": 2.5, "3-4": 3.5, ">4": 5.0},
    "weekly_online_learning_hours": {"<2": 1.0, "2-5": 3.5, "5-10": 7.5, ">10": 12.0},
    "digital_engagement_change": {"Decreased significantly": 1, "Decreased": 2, "Same": 3,
                                  "Increased": 4, "Increased significantly": 5},
    "missed_assessment_frequency": {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Very often": 5},
    "assessment_performance_trend": {"Significantly declining": 1, "Declining": 2, "Stable": 3,
                                     "Improving": 4, "Significantly improving": 5},
    "overall_academic_situation_change": {"Declined significantly": 1, "Declined": 2, "Same": 3,
                                          "Improved": 4, "Improved significantly": 5},
    "additional_academic_support_needed": {"No": 0, "Not sure": 1, "Yes": 2},
}
NOMINAL = ["programme", "department"]
TEXT_FIELDS = ["biggest_academic_challenge", "required_academic_support"]

LIKERT = [
    "assignment_completion_consistency", "class_attendance_frequency", "class_participation",
    "doubt_resolution_frequency", "faculty_academic_interaction", "study_consistency",
    "exam_cramming_frequency", "self_learning_frequency", "revision_frequency",
    "additional_resource_usage", "lms_usage_frequency", "online_learning_material_usage",
    "online_activity_completion", "late_assignment_frequency", "test_preparation_consistency",
    "course_interest", "academic_motivation", "career_goal_clarity",
    "academic_progress_satisfaction", "overall_academic_engagement",
    "programme_completion_confidence", "peer_support", "faculty_support",
    "academic_help_accessibility", "academic_guidance_frequency",
    "academic_workload_difficulty", "academic_performance_concern",
    "programme_completion_certainty",
]

VALID_RANGES = {
    "previous_sem_sgpa": (0.0, 10.0), "current_cgpa": (0.0, 10.0),
    "internal_assessment_average": (10.0, 100.0),  # <10 % is treated as a typo (e.g. "7%")
    "attendance_percentage": (0.0, 100.0), "courses_completed": (0, 60),
}

# 15-feature modelling set (README section 4).
FEATURES = [
    "previous_sem_sgpa", "current_cgpa", "internal_assessment_average",
    "attendance_percentage", "active_backlogs", "daily_study_hours",
    "study_consistency", "revision_frequency", "assignment_completion_consistency",
    "academic_motivation", "overall_academic_engagement", "academic_workload_difficulty",
    "academic_performance_concern", "late_assignment_frequency",
    "programme_completion_confidence",
]
TARGET = "at_risk"
REG_TARGET = "end_sem_sgpa"  # outcome of the in-progress semester (joined from results)

STATE_NAMES = ["Struggling", "Stable", "Strong"]
BAND_NAMES = ["Poor", "Average", "Good"]
BAND_EDGES = (6.5, 8.0)  # SGPA < 6.5 Poor, < 8.0 Average, else Good
ATTENDANCE_RULE = 75.0   # SRM minimum attendance for exam eligibility

PALETTE = {  # "Blackboard": board green, chalk white, one chalk-yellow highlight (dashboard + deck)
    "board": "#0f1a16", "panel": "#15231e", "chalk": "#ecebe4", "dust": "#8fa197", "highlight": "#e3c35a",
    "series": ["#b3861a", "#3a8ad0", "#cf4b44", "#8f6fd0", "#2f9a74"],   # CVD-validated, fixed order
    "states": ["#cf4b44", "#b3861a", "#3a8ad0"],                         # Struggling · Stable · Strong
}
