from rest_framework import serializers
from opint_framework.core.models import Action, IssueCategory, Solution


class ActionSerializer(serializers.ModelSerializer):

    # TODO: Check if those two are needed
    id = serializers.ReadOnlyField()
    action = serializers.CharField(required=True)

    class Meta:
        model = Action
        fields = ['id', 'action', 'last_modified']

    # Overwriting create to implement get_or_create approach in POST requests.
    def create(self, validated_data):
        # TODO: The line below relies in the fact that validated_data['action'] is either an id or an existing Action.
        # TODO: Will break if it's neither.
        print(validated_data['action'], validated_data['action'].isdigit())
        action = validated_data['action'] if not validated_data['action'].isdigit() else Action.objects.filter(pk=validated_data['action']).first().action
        guest, created = Action.objects.get_or_create(action=action)
        return guest


class IssueCategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = IssueCategory
        fields = ['id', 'regex', 'last_modified']


class SolutionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Solution
        fields = ['id', 'solution', 'propability', 'score', 'affected_site', 'last_modified']
