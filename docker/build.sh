git clone git@github.com:WildbookOrg/dev.git
docker build --target com.wildme.wildbook-image-curation.test --tag wildme/wildbook-image-curation-test .
docker build --target com.wildme.wildbook-image-curation.deploy --tag wildme/wildbook-image-curation .
# sudo port install socat
# socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\" &
# docker run -e DISPLAY=docker.for.mac.host.internal:0 wildme/wildbook-image-curation-test
# docker run -p 5000:5000 wildme/wildbook-image-curation
