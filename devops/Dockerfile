FROM wildme/wbia-provision:latest as org.wildme.wbia.build

MAINTAINER Wild Me <dev@wildme.org>

ARG BRANCH=next
ARG AZURE_DEVOPS_CACHEBUSTER=0

RUN echo "ARGS BRANCH=${BRANCH}"
RUN echo "ARGS AZURE_DEVOPS_CACHEBUSTER=${AZURE_DEVOPS_CACHEBUSTER}"

RUN git config --global user.email "dev@wildme.org" \
 && git config --global user.name "Wild Me Development Nightly Build (Azure CI)"

# TODO Temporarily disabled: Checkout latest version of all other repos
#RUN cd /wbia/wbia \
# && /virtualenv/env3/bin/python super_setup.py pull \
# && /virtualenv/env3/bin/python super_setup.py pull

# Remove all git association from the code
RUN find /wbia -name '.git' -type d -print0 | xargs -0 rm -rf

RUN /virtualenv/env3/bin/pip uninstall -y opencv-python opencv-python-headless

##########################################################################################
FROM wildme/wbia-base:latest as org.wildme.wbia.install

COPY --from=org.wildme.wbia.build /virtualenv /virtualenv

COPY --from=org.wildme.wbia.build /wbia /wbia

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

RUN /virtualenv/env3/bin/python /wbia/wildbook-ia/wbia/dev.py --set-workdir /data

##########################################################################################
FROM org.wildme.wbia.install as org.wildme.wbia.deploy

ENTRYPOINT ["/virtualenv/env3/bin/python", "/wbia/wildbook-ia/wbia/dev.py", "--dbdir", "/data/db", "--logdir", "/data/logs", "--web", "--port", "5000", "--web-deterministic-ports", "--containerized", "--cpudark", "--production"]

# Ports for the frontend web server
EXPOSE 5000

HEALTHCHECK --interval=2m --timeout=2m --retries=10 --start-period=1h  \
  CMD /virtualenv/env3/bin/python /bin/healthcheck.py

LABEL autoheal=true

STOPSIGNAL SIGTERM

##########################################################################################
FROM org.wildme.wbia.deploy as org.wildme.wbia.configure

CMD []
