FROM wildme.azurecr.io/ibeis/provision:latest as org.wildme.ibeis.build

MAINTAINER Wild Me <dev@wildme.org>

ARG BRANCH=next
ARG AZURE_DEVOPS_CACHEBUSTER=0

RUN echo "ARGS BRANCH=${BRANCH}"
RUN echo "ARGS AZURE_DEVOPS_CACHEBUSTER=${AZURE_DEVOPS_CACHEBUSTER}"

RUN git config --global user.email "dev@wildme.org" \
 && git config --global user.name "Wild Me Development Nightly Build (Azure CI)"

# Checkout latest version of all other repos
RUN cd /ibeis/ibeis \
 && /virtualenv/env3/bin/python super_setup.py --pull \
 && /virtualenv/env3/bin/python super_setup.py --pull

RUN cd /ibeis/ibeis-curvrank-module/ibeis_curvrank/ \
 && rm -rf __init__.py \
 && git pull \
 && cd /ibeis/ibeis-kaggle7-module/ibeis_kaggle7/ \
 && git pull

RUN cd /ibeis/ibeis \
 && git checkout ${BRANCH} \
 && git pull \
 && cd /ibeis/ibeis_cnn \
 && git checkout ${BRANCH} \
 && git pull

RUN cd /ibeis/detecttools \
 && rm -rf .git \
 && cd /ibeis/dtool \
 && rm -rf .git \
 && cd /ibeis/ibeis_cnn \
 && rm -rf .git \
 && cd /ibeis/guitool \
 && rm -rf .git \
 && cd /ibeis/hesaff \
 && rm -rf .git \
 && cd /ibeis/lightnet \
 && rm -rf .git \
 && cd /ibeis/brambox \
 && rm -rf .git \
 && cd /ibeis/ibeis-flukematch-module \
 && rm -rf .git \
 && cd /ibeis/ibeis-curvrank-module/ibeis_curvrank \
 && rm -rf .git \
 && cd /ibeis/ibeis-curvrank-module \
 && rm -rf .git \
 && cd /ibeis/ibeis-deepsense-module \
 && rm -rf .git \
 && cd /ibeis/ibeis-finfindr-module \
 && rm -rf .git \
 && cd /ibeis/ibeis-kaggle7-module/ibeis_kaggle7 \
 && rm -rf .git \
 && cd /ibeis/ibeis-kaggle7-module \
 && rm -rf .git \
 && cd /ibeis/ibeis-2d-orientation-module/ibeis_2d_orientation \
 && rm -rf .git \
 && cd /ibeis/ibeis-2d-orientation-module \
 && rm -rf .git \
 && cd /ibeis/plottool \
 && rm -rf .git \
 && cd /ibeis/pydarknet \
 && rm -rf .git \
 && cd /ibeis/pyrf \
 && rm -rf .git \
 && cd /ibeis/ubelt \
 && rm -rf .git \
 && cd /ibeis/utool \
 && rm -rf .git \
 && cd /ibeis/vtool \
 && rm -rf .git \
 && cd /ibeis/ibeis \
 && rm -rf .git

RUN /virtualenv/env3/bin/pip uninstall -y opencv-python opencv-python-headless

##########################################################################################
FROM wildme.azurecr.io/ibeis/base:latest as org.wildme.ibeis.install

COPY --from=org.wildme.ibeis.build /virtualenv /virtualenv

COPY --from=org.wildme.ibeis.build /ibeis /ibeis

RUN ln -s /virtualenv/env3/lib/libgpuarray.so     /usr/lib/libgpuarray.so \
 && ln -s /virtualenv/env3/lib/libgpuarray.so.3   /usr/lib/libgpuarray.so.3 \
 && ln -s /virtualenv/env3/lib/libgpuarray.so.3.0 /usr/lib/libgpuarray.so.3.0

# Add theano configuration file
ADD ./provision/_config/theanorc /root/.theanorc

# Add Boto configuration file for access to AWS S3 storage with this container
ADD ./provision/_config/boto.cfg /etc/boto.cfg

# Add embed script for python development
ADD ./provision/_config/embed.sh /bin/embed

# Add Python health check
ADD ./provision/_config/healthcheck.py /bin/healthcheck.py

RUN mkdir /data

VOLUME /data

RUN /virtualenv/env3/bin/python /ibeis/ibeis/dev.py --set-workdir /data

##########################################################################################
FROM org.wildme.ibeis.install as org.wildme.ibeis.deploy

ENTRYPOINT ["/virtualenv/env3/bin/python", "/ibeis/ibeis/dev.py", "--dbdir", "/data/docker", "--logdir", "/data/docker/_ibsdb/_ibeis_logs/", "--web", "--port", "5000", "--web-deterministic-ports", "--containerized", "--cpudark", "--production"]

# Ports for the frontend web server
EXPOSE 5000

HEALTHCHECK --interval=2m --timeout=2m --retries=10 --start-period=1h  \
  CMD /virtualenv/env3/bin/python /bin/healthcheck.py

LABEL autoheal=true

STOPSIGNAL SIGTERM

##########################################################################################
FROM org.wildme.ibeis.deploy as org.wildme.ibeis.configure

CMD []
