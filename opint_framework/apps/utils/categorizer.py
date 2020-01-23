import difflib

from opint_framework.apps.core.models import IssueCategory


def categorize_issue(issue):
    categories = IssueCategory.objects.all()
    best_match_similarity = 0
    match_category = None
    for category in categories:
        similarity = difflib.SequenceMatcher(None, issue.message, category.regex).ratio()
        if similarity > 0.95 and similarity > best_match_similarity:
            best_match_similarity = similarity
            match_category = category
    return match_category
