Run WBIA
```
docker run -p 85:5000 wildme/wbia:latest bash 
```

Run schemathesis
```
cd wildbook-ia/api_docs/schemathesis
uvx schemathesis run ../openapi/api_manual.yaml --url http://localhost:85 > logs/api_manual.txt 2>&1
uvx schemathesis run ../openapi/api_claude.yaml --url http://localhost:85 > logs/api_claude.txt 2>&1
```