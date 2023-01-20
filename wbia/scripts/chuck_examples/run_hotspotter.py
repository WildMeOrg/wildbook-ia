'''
Run HOTSPOTTER (LNBNN) given a list of query and database 
annotation ids.

1. For efficiency reasons that depend on the way the 
   search data structure is currently built, it is best
   to not change the daids list. The search structure is
   rebuilt each time a new daids list is provided.
   The first call, therefore, is usually very slow.

2. Each run that query_chips_graph method call takes a
   list of aids that are assumed to be from the same
   animal - e.g. from a single encounter. Therefore,
   if no information is given about an annotation it
   should be queried as a singleton list.

3. The method called here builds a lot of cached
   information about the potential matching annotations
   for the purposes of visualization (thumbnail images of
   matches, for example). This slows the function down
   considerably. There are other calls, such as
   query_chips that do much less work and are therefore
   much faster.
'''


def run_hotspotter(ibs, qaids_list, daids):
    # This query configuration changes for each
    # algorithm run.
    query_config = {
        # HotSpotter LNBNN
        'K': 5,
        'Knorm': 5,

        # HotSpotter Background Subtraction
        'fg_on': True,
        'prescore_method': 'csum',

        # HotSpotter Spatial Verification
        'sv_on': True,

        # HotSpotter Aggregation
        'can_match_sameimg': False,
        'can_match_samename': True,
        'score_method': 'csum',
    }

    # Run Hotspotter on each qaid list against
    # the database. Store the summary results 
    # for output at the end.
    matches_list = []
    for qaids in qaids_list:
        query_result = ibs.query_chips_graph(
            qaid_list=qaids,
            daid_list=daids,
            query_config_dict=query_config,
            echo_query_params=False
        )

        annot_matches = query_result['summary_annot']
        name_matches = query_result['summary_name']
        matches = annot_matches + name_matches
        matches.sort(key=lambda match: match.get('score'), reverse=True)
        matches_list.append(matches)

    # Output the results.
    for qaids, matches in zip(qaids_list, matches_list):
        print('=' * 25)
        seen = set([])
        for match in matches[:10]:  # only show the top 10
            score = match['score']
            daid = match['daid']
            dnid = match['dnid']
            if daid in seen:
                continue
            print(f'Query AIDs = {qaids}, Database AID = {daid}, Database NID = {dnid} (Score: {score:0.02f})')
            seen.add(daid)

# aids = ibs.get_valid_aids()
# aids = ca_aids
qaids_list = [[aids[0]], [aids[1]], [aids[2]]]
daids = aids[:]

run_hotspotter(ibs, qaids_list, daids)
