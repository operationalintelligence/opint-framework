from django.contrib import admin

from .models import WorkflowIssue


class WorkflowIssueAdmin(admin.ModelAdmin):
    list_display = ['id', 'message', 'category', 'status', 'last_modified']


admin.site.register(WorkflowIssue, WorkflowIssueAdmin)
