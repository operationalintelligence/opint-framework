from django.contrib import admin

from .models import IssueCategory, Action, Solution


class IssueCategoryAdmin(admin.ModelAdmin):
    list_display = ['id', 'regex', 'amount', 'last_modified']


class ActionAdmin(admin.ModelAdmin):
    list_display = ['id', 'action', 'last_modified']


class SolutionAdmin(admin.ModelAdmin):
    list_display = ['id', 'solution', 'propability', 'score', 'last_modified', 'affected_site']


admin.site.register(IssueCategory, IssueCategoryAdmin)
admin.site.register(Action, ActionAdmin)
admin.site.register(Solution, SolutionAdmin)
