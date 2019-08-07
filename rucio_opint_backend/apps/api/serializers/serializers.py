from rest_framework import serializers
from rucio_opint_backend.apps.core.models import Issue, Action, IssueCause, IssueCategory, Solution


class IssueSerializer(serializers.ModelSerializer):
    class Meta:
        model = Issue
        fields = ['id', 'message', 'src_site', 'dst_site', 'category', 'amount', 'type', 'status', 'last_modified']


class ActionSerializer(serializers.ModelSerializer):

    # TODO: Check if those two are needed
    id = serializers.ReadOnlyField()
    action = serializers.CharField(required=True)

    class Meta:
        model = Action
        fields = ['id', 'action', 'last_modified']

    # Overwriting create to implement get_or_create approach in POST requests.
    def create(self, validated_data):
        # TODO: The line below relys in the fact that validated_data['action'] is either an id or an existing Action.
        # TODO: Will break if it's neither.
        print(validated_data['action'], validated_data['action'].isdigit())
        guest, created = Action.objects.get_or_create(action=validated_data['action'] if not validated_data['action'].isdigit()
        else Action.objects.filter(pk=validated_data['action']).first().action)
        return guest

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
