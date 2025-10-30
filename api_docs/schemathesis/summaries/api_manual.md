# Summary of schemathesis failures for the manual API 

## Overview 
See log here: https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt

API Operations:
- Selected: 19/19
- Tested: 19

Failures: 
- Server error: 21
- API accepted schema-violating request: 4
- API rejected schema-compliant request: 5
- Unsupported methods: 2

## Failure summaries 
### (1) No validation of input parameters - 23 errors

Errors: 
- Server error (18)
- API accepted schema-violating request (5)

Test Cases: 
1. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L48: `/api/engine/detect/cnn/yolo/` accepts `jobid` of the wrong type ({})
2. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L64: `/api/engine/detect/cnn/yolo/` accepts invalid input and then crashes with type error resulting in server error (500) rather than Missing or Invalid Input Error (400)
3. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L94: `/api/engine/query/graph/complete/` accepts `jobid` of the wrong type and then crashes with assertion error resulting in server error (500) rather than Missing or Invalid Input Error (400)
4. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L106: `/api/engine/query/graph/complete/` accepts requestbody of the wrong type ([]).
5. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L139: `/api/engine/query/graph/complete/` accepts empty GET parameter then crashes with assertion error resulting in server error (500) rather than Missing or Invalid Input Error (400)
6. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L169: `/api/engine/query/graph/` accepts invalid requestbody then crashes with assertion error resulting in server error (500) rather than Missing or Invalid Input Error (400)
7. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L199: `/api/engine/query/graph/` accepts invalid GET parameters then crashes with assertion error resulting in server error (500) rather than Missing or Invalid Input Error (400)
8. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L212: `/api/annot/json/` accepts invalid requestbody then crashes, resulting in server error (500) rather than Missing or Invalid Input Error (400)
9. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L225: `/api/annot/json/` DELETE accepts invalid UUIDs. `get_annot_aids_from_uuid` returns None for UUIDs that don't exist in the database. These None values are then passed to `delete_annots`, which causes the SQL error WHERE annotations_rowid IN ( None ). This causes a server error (500) rather than Missing or Invalid Input Error (400)
10. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L238: `/api/image/json/` DELETE accepts invalid UUIDs. `get_annot_aids_from_uuid` returns None for UUIDs that don't exist in the database. These None values are then passed to `delete_annots`, which causes the SQL error WHERE annotations_rowid IN ( None ). This causes a server error (500) rather than Missing or Invalid Input Error (400)
11. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L251: `/api/engine/detect/cnn/lightnet/` POST accepts `jobid` of the wrong type ({}) and then crashes with assertion error resulting in server error (500) rather than Missing or Invalid Input Error (400)
12. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L264: `/api/engine/detect/cnn/lightnet/` GET accepts `jobid` of the wrong type and then crashes with assertion error resulting in server error (500) rather than Missing or Invalid Input Error (400)
13. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L277: `/api/engine/detect/cnn/` POST accepts an invalid array of null values and tries to use it to update a dictionary, resulting in server error (500) rather than Missing or Invalid Input Error (400)
14. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L290: `/api/engine/detect/cnn/` GET accepts `jobid` of the wrong type and then crashes with value error resulting in server error (500) rather than Missing or Invalid Input Error (400)
15. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L303: `/api/engine/job/status/` POST accepts `jobid` of the wrong type ({}) and then returns success (200) with response status being "error", rather than Missing or Invalid Input Error (400)
16. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L320: `/api/engine/job/result/` POST accepts `jobid` of the wrong type ({}) and then returns success (200) with response status being "error", rather than Missing or Invalid Input Error (400)
17. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L336: `/api/engine/job/result/` POST accepts `jobid` of the wrong type ([]) and then crashes with internal server error (500) rather than Missing or Invalid Input Error (400)
18. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L349: `/api/engine/job/result/` GET accepts `jobid` of the wrong type ([]) and then crashes with internal server error (500) rather than Missing or Invalid Input Error (400)
19. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L362: `/api/annot/json/` GET (`get_valid_annot_uuids_json(ibs, **kwargs)`) accepts any parameters (via `**kwargs`), but then blindly passes them to `get_valid_aids()` which only accepts specific parameters, causing a crash with internal server error (500) rather than Missing or Invalid Input Error (400)
20. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L375: `/api/image/json/` POST only validates GPS lat/lon when both are provided, resulting in success (200) rather than Missing or Invalid Input Error (400) when an invalid `image_gps_lon_list` is provided. 
21. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L391: `/api/image/json/` POST accepts `image_uri_list` of the wrong type ([]) and then crashes with internal server error (500) rather than Missing or Invalid Input Error (400)
22. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L404: `/api/image/json/` GET accepts invalid `imgsetid_list` and then crashes with internal server error (500) rather than Missing or Invalid Input Error (400)
23. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L417: `/api/engine/job/status/` GET accepts invalid garbage parameters from the fuzzing test and then crashes with internal server error (500) rather than Missing or Invalid Input Error (400)

### (2) `schemathesis` fails to recoginize custom 6XX error codes (LESS IMPORTANT) - 9 errors

Errors: 
- Server error (5)
- API rejected schema-compliant request (4)

Test Cases: 
1. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L31
2. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L77
3. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L122
4. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L152
5. https://github.com/WildMeOrg/wildbook-ia/blob/api/api_docs/schemathesis/logs/api_manual.txt#L182