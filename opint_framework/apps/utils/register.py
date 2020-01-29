import time

from opint_framework.core.models import TransferIssue, IssueCategory
from opint_framework.apps.utils.categorizer import categorize_issue


def register_transfer_issue(issue):
    print("INFO: registering issue ", issue)
    obj, created = TransferIssue.objects.get_or_create(message=issue.pop('message'),
                                                       src_site=issue.pop('src_site'),
                                                       dst_site=issue.pop('dst_site'),
                                                       type=issue.pop('type'),
                                                       defaults=issue)
    if not created:
        obj.last_modified = time.time()
        obj.save(update_fields=['last_modified'])
        # return
    category = categorize_issue(obj)
    if not category:
        print("INFO: creating new category")
        category = IssueCategory(regex=obj.message, amount=obj.amount)
        category.save()
    else:
        print("INFO: assigning to existing category")
        category.amount += obj.amount
        category.save(update_fields=['amount'])

    obj.category = category
    obj.save(update_fields=['category'])
