from rest_framework import serializers
from rucio_opint_backend.apps.core.models import Issue, Action, IssueCause, IssueCategory, Solution


class IssueSerializer(serializers.ModelSerializer):
    class Meta:
        model = Issue
        fields = ['id', 'message', 'src_site', 'dst_site', 'category', 'amount', 'type', 'status', 'last_modified']


class ActionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Action
        fields = ['id', 'action', 'last_modified']


class IssueCauseSerializer(serializers.ModelSerializer):
    class Meta:
        model = IssueCause
        fields = ['id', 'cause', 'last_modified']


class IssueCategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = IssueCategory
        fields = ['id', 'amount', 'regex', 'cause', 'last_modified']


class SolutionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Solution
        fields = ['id', 'category', 'proposed_action', 'solution', 'real_cause', 'propability',
                  'score', 'affected_site', 'last_modified']
