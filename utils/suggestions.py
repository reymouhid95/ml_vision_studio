"""
Suggestions automatiques basées sur la qualité des données et les résultats d'entraînement.

Appelé après chaque phase d'entraînement pour guider l'utilisateur.
"""
from __future__ import annotations


def analyze_class_balance(classes: list[dict]) -> list[str]:
    """Suggestions basées sur la distribution des classes avant/après collecte."""
    if not classes:
        return []

    counts = [len(c["samples"]) for c in classes]
    names  = [c["name"] for c in classes]
    total  = sum(counts)
    n_cls  = len(classes)

    suggestions: list[str] = []

    # Exemples insuffisants
    for name, count in zip(names, counts):
        if count < 5:
            suggestions.append(
                f"⚠️  « {name} » : {count} exemple(s) seulement — minimum recommandé 10"
            )
        elif count < 10:
            suggestions.append(
                f"💡 « {name} » : {count} exemples — en ajouter davantage améliorera la précision"
            )

    # Déséquilibre entre classes
    if counts:
        mn, mx = min(counts), max(counts)
        if mx > 0 and mn > 0 and mx / mn >= 3:
            min_name = names[counts.index(mn)]
            max_name = names[counts.index(mx)]
            suggestions.append(
                f"⚠️  Déséquilibre : « {max_name} » ({mx} ex.) vs « {min_name} » ({mn} ex.) "
                "— équilibrez les classes pour de meilleurs résultats"
            )

    # Volume global
    if total < 20 * n_cls:
        suggestions.append(
            f"💡 {total} exemples au total ({total // n_cls} moy./classe) — "
            "20+ exemples/classe donnent des résultats plus fiables"
        )

    return suggestions


def analyze_training_results(
    loss_hist: list[float],
    *,
    train_acc: float | None = None,
    val_acc: float | None = None,
    n_classes: int = 2,
) -> list[str]:
    """Suggestions post-entraînement à partir des métriques de training."""
    suggestions: list[str] = []

    # Surapprentissage
    if train_acc is not None and val_acc is not None:
        gap = train_acc - val_acc
        if gap > 0.20:
            suggestions.append(
                f"⚠️  Surapprentissage : acc_train={train_acc:.1%} vs acc_val={val_acc:.1%} "
                f"(écart {gap:.1%}) — ajoutez des données ou augmentez le dropout"
            )

    # Précision proche du hasard
    ref_acc = 1 / n_classes
    check_acc = val_acc if val_acc is not None else train_acc
    if check_acc is not None and check_acc < ref_acc + 0.05:
        suggestions.append(
            f"⚠️  Précision proche du hasard ({check_acc:.1%} pour {n_classes} classes) — "
            "vérifiez la qualité et la diversité des données"
        )

    # Stagnation de la perte
    if len(loss_hist) >= 10:
        last = loss_hist[-5:]
        first_last = loss_hist[-10:-5]
        if first_last and last:
            avg_first = sum(first_last) / len(first_last)
            avg_last  = sum(last)       / len(last)
            if avg_last > avg_first * 0.98:
                suggestions.append(
                    "💡 La perte a stagné sur les dernières époques — "
                    "essayez un taux d'apprentissage plus élevé ou plus d'époques"
                )

    # Succès
    if check_acc is not None and check_acc >= 0.95:
        suggestions.append(f"✅ Excellente précision ({check_acc:.1%}) — modèle prêt à utiliser !")
    elif check_acc is not None and check_acc >= 0.80:
        suggestions.append(f"✅ Bonne précision ({check_acc:.1%}) — résultats solides")

    return suggestions


def format_suggestions(items: list[str]) -> str:
    return "\n".join(items) if items else "✓ Aucun problème détecté."
