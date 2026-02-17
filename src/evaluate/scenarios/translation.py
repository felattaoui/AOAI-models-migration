"""
Translation evaluation scenario.

Tests whether models maintain translation quality across migration.
Provides multilingual test pairs (FR/EN/DE) with reference translations.

Metrics:
- fluency: Is the translation natural and well-written?
- coherence: Does the translation maintain logical structure?
- relevance: Does the translation accurately convey the source meaning?
  (Used as a proxy for semantic equivalence via LLM-as-Judge)
"""

from src.evaluate.core import TestCase, MigrationEvaluator
from src.evaluate.prompts import load_prompty


TRANSLATION_METRICS = ["fluency", "coherence", "relevance"]


# ---------------------------------------------------------------------------
# System prompts loaded from .prompty files
# ---------------------------------------------------------------------------

TRANSLATE_FR_TO_EN = load_prompty("translate_fr_en")["system_prompt"]
TRANSLATE_EN_TO_FR = load_prompty("translate_en_fr")["system_prompt"]
TRANSLATE_EN_TO_DE = load_prompty("translate_en_de")["system_prompt"]
TRANSLATE_TECHNICAL = load_prompty("translate_technical")["system_prompt"]


# ---------------------------------------------------------------------------
# Sample test cases with reference translations
# ---------------------------------------------------------------------------

TRANSLATION_TEST_CASES = [
    # 1. Formal business French → English
    TestCase(
        prompt="Nous avons le plaisir de vous informer que votre candidature a été retenue pour le poste de Directeur des Opérations. Votre prise de fonction est prévue pour le 1er avril 2026. Veuillez trouver ci-joint les détails de votre contrat ainsi que les modalités d'intégration.",
        system_prompt=TRANSLATE_FR_TO_EN,
        expected_output="We are pleased to inform you that your application has been selected for the position of Director of Operations. Your start date is scheduled for April 1st, 2026. Please find attached the details of your contract and onboarding arrangements.",
        metadata={"direction": "FR→EN", "domain": "business", "difficulty": "standard"},
    ),
    # 2. Technical documentation English → French
    TestCase(
        prompt="The API endpoint accepts POST requests with a JSON body containing the `user_id` and `action` fields. Authentication is handled via Bearer tokens in the Authorization header. Rate limiting is set to 1000 requests per minute per API key.",
        system_prompt=TRANSLATE_TECHNICAL,
        expected_output="Le point de terminaison API accepte les requêtes POST avec un corps JSON contenant les champs `user_id` et `action`. L'authentification est gérée via des jetons Bearer dans l'en-tête Authorization. La limitation de débit est fixée à 1000 requêtes par minute par clé API.",
        metadata={"direction": "EN→FR", "domain": "technical", "difficulty": "standard"},
    ),
    # 3. Idiomatic French → English (challenging)
    TestCase(
        prompt="Il ne faut pas mettre la charrue avant les boeufs. Commençons par valider le concept avant de parler de mise en production. On verra bien si ça tient la route une fois qu'on aura les premiers retours terrain.",
        system_prompt=TRANSLATE_FR_TO_EN,
        expected_output="We shouldn't put the cart before the horse. Let's start by validating the concept before talking about going to production. We'll see if it holds up once we get the initial field feedback.",
        metadata={"direction": "FR→EN", "domain": "informal/idiomatic", "difficulty": "hard"},
    ),
    # 4. Marketing content English → French
    TestCase(
        prompt="Unlock the full potential of your data with our AI-powered analytics platform. Get actionable insights in real-time, reduce operational costs by up to 40%, and make data-driven decisions with confidence. Start your free 30-day trial today.",
        system_prompt=TRANSLATE_EN_TO_FR,
        expected_output="Libérez tout le potentiel de vos données grâce à notre plateforme d'analyse alimentée par l'IA. Obtenez des insights exploitables en temps réel, réduisez vos coûts opérationnels jusqu'à 40 % et prenez des décisions basées sur les données en toute confiance. Commencez votre essai gratuit de 30 jours dès aujourd'hui.",
        metadata={"direction": "EN→FR", "domain": "marketing", "difficulty": "standard"},
    ),
    # 5. Legal text French → English
    TestCase(
        prompt="Conformément à l'article L.121-20-2 du Code de la consommation, le droit de rétractation ne peut être exercé pour les contrats de fourniture de services pleinement exécutés avant la fin du délai de rétractation et dont l'exécution a commencé après accord préalable exprès du consommateur.",
        system_prompt=TRANSLATE_FR_TO_EN,
        expected_output="In accordance with Article L.121-20-2 of the Consumer Code, the right of withdrawal cannot be exercised for contracts for the provision of services fully performed before the end of the withdrawal period and whose performance began after the express prior consent of the consumer.",
        metadata={"direction": "FR→EN", "domain": "legal", "difficulty": "hard"},
    ),
    # 6. Conversational English → German
    TestCase(
        prompt="Hey team, just a quick heads up - the deployment is scheduled for tonight at 11 PM. Please make sure your PRs are merged by 5 PM. If you run into any blockers, ping me on Slack. Let's make this a smooth release!",
        system_prompt=TRANSLATE_EN_TO_DE,
        expected_output="Hey Team, nur ein kurzer Hinweis - das Deployment ist für heute Nacht um 23 Uhr geplant. Bitte stellt sicher, dass eure PRs bis 17 Uhr gemergt sind. Falls ihr auf Hindernisse stoßt, schreibt mir auf Slack. Lasst uns ein reibungsloses Release hinlegen!",
        metadata={"direction": "EN→DE", "domain": "informal/tech", "difficulty": "standard"},
    ),
    # 7. Nuanced text with cultural references FR → EN
    TestCase(
        prompt="La rentrée approche et avec elle son lot de réunions de cadrage, de séminaires de cohésion d'équipe et de points d'étape trimestriels. C'est aussi le moment de revoir nos objectifs S2 et de préparer le budget prévisionnel 2027.",
        system_prompt=TRANSLATE_FR_TO_EN,
        expected_output="The back-to-school season is approaching, and with it the usual round of scoping meetings, team-building seminars, and quarterly review sessions. It's also time to revisit our H2 objectives and prepare the 2027 forecast budget.",
        metadata={"direction": "FR→EN", "domain": "business/cultural", "difficulty": "hard"},
    ),
    # 8. Error message / UI text English → French (short, precise)
    TestCase(
        prompt="Unable to process your request. The file exceeds the maximum upload size of 25 MB. Please compress your file or split it into smaller parts and try again.",
        system_prompt=TRANSLATE_EN_TO_FR,
        expected_output="Impossible de traiter votre demande. Le fichier dépasse la taille maximale de téléversement de 25 Mo. Veuillez compresser votre fichier ou le diviser en parties plus petites et réessayer.",
        metadata={"direction": "EN→FR", "domain": "UI/UX", "difficulty": "easy"},
    ),
    # 9. Medical/scientific English → French
    TestCase(
        prompt="The Phase III clinical trial demonstrated a statistically significant reduction in HbA1c levels (p<0.001) in patients receiving the combination therapy versus monotherapy. The most common adverse events were mild gastrointestinal symptoms (nausea: 12%, diarrhea: 8%).",
        system_prompt=TRANSLATE_EN_TO_FR,
        expected_output="L'essai clinique de Phase III a démontré une réduction statistiquement significative des taux d'HbA1c (p<0,001) chez les patients recevant la thérapie combinée par rapport à la monothérapie. Les événements indésirables les plus fréquents étaient des symptômes gastro-intestinaux légers (nausées : 12 %, diarrhée : 8 %).",
        metadata={"direction": "EN→FR", "domain": "medical", "difficulty": "hard"},
    ),
    # 10. Ambiguous sentence (tests interpretation)
    TestCase(
        prompt="Les poules du couvent couvent.",
        system_prompt=TRANSLATE_FR_TO_EN,
        expected_output="The hens of the convent are brooding.",
        metadata={"direction": "FR→EN", "domain": "ambiguous/homograph", "difficulty": "tricky"},
    ),
]


def create_translation_evaluator(
    source_model: str = "gpt-4o",
    target_model: str = "gpt-4.1",
    test_cases: list[TestCase] | None = None,
    **kwargs,
) -> MigrationEvaluator:
    """
    Create a pre-configured evaluator for translation scenarios.

    Args:
        source_model: Current model to migrate from.
        target_model: New model to migrate to.
        test_cases: Custom test cases (uses built-in examples if None).

    Returns:
        Configured MigrationEvaluator ready to run.

    Example:
        evaluator = create_translation_evaluator("gpt-4o", "gpt-4.1")
        report = evaluator.run()
        report.print_report()
    """
    return MigrationEvaluator(
        source_model=source_model,
        target_model=target_model,
        test_cases=test_cases or TRANSLATION_TEST_CASES,
        metrics=TRANSLATION_METRICS,
        **kwargs,
    )
