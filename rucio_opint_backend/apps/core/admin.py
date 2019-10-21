from django.contrib import admin

from .models import WorkflowIssue, TransferIssue, IssueCategory, IssueCause, Action, Solution


class WorkflowIssueAdmin(admin.ModelAdmin):
    list_display = ['id', 'message', 'workflow', 'category', 'type', 'status', 'last_modified']


class TransferIssueAdmin(admin.ModelAdmin):
    list_display = ['id', 'message', 'src_site', 'dst_site', 'category', 'type', 'status', 'last_modified']


class IssueCategoryAdmin(admin.ModelAdmin):
    list_display = ['id', 'regex', 'cause', 'amount', 'last_modified']


class IssueCauseAdmin(admin.ModelAdmin):
    list_display = ['id', 'cause', 'last_modified']


class ActionAdmin(admin.ModelAdmin):
    list_display = ['id', 'action', 'last_modified']


class SolutionAdmin(admin.ModelAdmin):
    list_display = ['id', 'category', 'proposed_action', 'solution', 'real_cause',
                    'propability', 'score', 'last_modified', 'affected_site']


admin.site.register(WorkflowIssue, WorkflowIssueAdmin)
admin.site.register(TransferIssue, TransferIssueAdmin)
admin.site.register(IssueCategory, IssueCategoryAdmin)
admin.site.register(IssueCause, IssueCauseAdmin)
admin.site.register(Action, ActionAdmin)
admin.site.register(Solution, SolutionAdmin)
