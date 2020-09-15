FROM wildme/wbia-provision:latest as org.wildme.wbia.build

MAINTAINER Wild Me <dev@wildme.org>

ARG BRANCH=next

RUN echo "ARGS BRANCH=${BRANCH}"

RUN git config --global user.email "dev@wildme.org" \
 && git config --global user.name "Wild Me Development Nightly Build (Azure CI)"

# Remove all git association from the code
RUN find /wbia -name '.git' -type d -print0 | xargs -0 rm -rf

RUN set -ex \
 && /virtualenv/env3/bin/pip uninstall -y opencv-python opencv-python-headless \
 && /virtualenv/env3/bin/pip uninstall -y dtool-ibeis guitool-ibeis plottool-ibeis utool-ibeis vtool-ibeis \
 && /virtualenv/env3/bin/pip uninstall -y dtool guitool plottool utool vtool \
 && /virtualenv/env3/bin/pip uninstall -y tensorflow tensorflow-estimator tensorboard tensorboard-plugin-wit keras

RUN set -ex \
 && /virtualenv/env3/bin/pip install tensorflow-gpu==1.15 keras==2.2.5

##########################################################################################
FROM wildme/wbia-base:latest as org.wildme.wbia.install

MAINTAINER Wild Me <dev@wildme.org>

COPY --from=org.wildme.wbia.build /virtualenv /virtualenv

COPY --from=org.wildme.wbia.build /wbia /wbia

RUN set -ex \
 && ln -s /virtualenv/env3/lib/libgpuarray.so     /usr/lib/libgpuarray.so \
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

RUN /virtualenv/env3/bin/python /wbia/wildbook-ia/wbia/dev.py --set-workdir /data

##########################################################################################
FROM org.wildme.wbia.install as org.wildme.wbia.deploy

ENTRYPOINT ["/virtualenv/env3/bin/python", "-m", "wbia" "--dbdir", "/data/db", "--logdir", "/data/logs", "--web", "--port", "5000", "--web-deterministic-ports", "--containerized", "--cpudark", "--production"]

# Ports for the frontend web server
EXPOSE 5000

HEALTHCHECK --interval=2m --timeout=2m --retries=10 --start-period=1h  \
  CMD /virtualenv/env3/bin/python /bin/healthcheck.py

LABEL autoheal=true

STOPSIGNAL SIGTERM

##########################################################################################
FROM org.wildme.wbia.deploy as org.wildme.wbia.configure

CMD []
