FROM wildme/wbia-provision:latest as org.wildme.wbia.build

# Grab latest code and remove all Git histories from code repositories
RUN set -ex \
 && git config --global user.email "dev@wildme.org" \
 && git config --global user.name "Wild Me" \
 && find /wbia/wbia* -name '.git' -type d -print0 | xargs -0 -i /bin/bash -c \
    'cd {} && cd .. && echo $(pwd) && (git stash && git pull && git stash pop || git reset --hard origin/develop)' \
 && cd /wbia/wbia-plugin-curvrank/wbia_curvrank \
 && git stash && git pull && git stash pop || git reset --hard origin/develop \
 && cd /wbia/wbia-plugin-kaggle7/wbia_kaggle7 \
 && git stash && git pull && git stash pop || git reset --hard origin/develop \
 && find /wbia -name '.git' -type d -print0 | xargs -0 rm -rf \
 && find /wbia -name '_skbuild' -type d -print0 | xargs -0 rm -rf

RUN /virtualenv/env3/bin/pip uninstall -y opencv-python opencv-python-headless

RUN /virtualenv/env3/bin/pip uninstall -y dtool-ibeis guitool-ibeis plottool-ibeis utool-ibeis vtool-ibeis \
 && /virtualenv/env3/bin/pip uninstall -y dtool guitool plottool utool vtool

##########################################################################################
FROM wildme/wbia-base:latest as org.wildme.wbia.install

LABEL autoheal=true

COPY --from=org.wildme.wbia.build /virtualenv /virtualenv

COPY --from=org.wildme.wbia.build /wbia /wbia

# Add theano configuration file
ADD ./_config/theanorc /root/.theanorc

# Add embed script for python development
ADD ./_config/embed.sh /bin/embed

# Add Python health check
ADD ./_config/healthcheck.py /bin/healthcheck.py

RUN mkdir /data

VOLUME /data

RUN /virtualenv/env3/bin/python /wbia/wildbook-ia/wbia/dev.py --set-workdir /data

##########################################################################################
FROM org.wildme.wbia.install as org.wildme.wbia.deploy

ENTRYPOINT ["/virtualenv/env3/bin/python", "/wbia/wildbook-ia/wbia/dev.py", "--dbdir", "/data/db", "--logdir", "/data/logs", "--web", "--port", "5000", "--web-deterministic-ports", "--containerized", "--cpudark", "--production"]

# Ports for the frontend web server
EXPOSE 5000

HEALTHCHECK --interval=2m --timeout=2m --retries=10 --start-period=1h \
  CMD /virtualenv/env3/bin/python /bin/healthcheck.py

STOPSIGNAL SIGTERM

##########################################################################################
FROM org.wildme.wbia.deploy as org.wildme.wbia.configure

CMD []
