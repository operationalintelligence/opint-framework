from django.contrib import admin

from .models import TransferIssue


class TransferIssueAdmin(admin.ModelAdmin):
    list_display = ['id', 'message', 'src_site', 'dst_site', 'category', 'type', 'status', 'last_modified']


admin.site.register(TransferIssue, TransferIssueAdmin)
