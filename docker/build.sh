git clone git@github.com:WildbookOrg/dev.git
docker build --target com.wildme.wildbook-image-curation.test --tag wildbook/image-curation-test .
docker run -ti --rm \
       -e DISPLAY=$DISPLAY \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       wildbook/image-curation-test:latest
docker build --target com.wildme.wildbook-image-curation.deploy --tag wildbook/image-curation .

