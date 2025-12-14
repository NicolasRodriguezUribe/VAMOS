from vamos.ux.analytics.genealogy import GenealogyTracker, get_lineage, compute_operator_success_stats, compute_generation_contributions


def test_genealogy_tracker_lineage_and_stats():
    tracker = GenealogyTracker()
    p1 = tracker.new_individual(generation=0, parents=[], operator_name=None, algorithm_name=None, fitness=None)
    p2 = tracker.new_individual(generation=0, parents=[], operator_name=None, algorithm_name=None, fitness=None)
    c1 = tracker.new_individual(generation=1, parents=[p1, p2], operator_name="opA", algorithm_name="algo", fitness=None)
    tracker.mark_final_front([c1])
    lineage = get_lineage(tracker, c1)
    ids = {rec.individual_id for rec in lineage}
    assert ids == {p1, p2, c1}
    df_ops = compute_operator_success_stats(tracker)
    assert not df_ops.empty
    df_gen = compute_generation_contributions(tracker)
    assert not df_gen.empty
