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
 && find /wbia -name '.git' -type d -print0 | xargs -0 rm -rf

# Run smoke tests
RUN set -ex \
 && /virtualenv/env3/bin/python -c "import wbia;            from wbia.__main__ import smoke_test; smoke_test()" \
 && /virtualenv/env3/bin/python -c "import wbia_cnn;        from wbia_cnn.__main__ import main;   main()" \
 && /virtualenv/env3/bin/python -c "import wbia_pie;        from wbia_pie.__main__ import main;   main()" \
 && /virtualenv/env3/bin/python -c "import wbia_flukematch; from wbia_flukematch.plugin import *" \
 && /virtualenv/env3/bin/python -c "import wbia_curvrank;   from wbia_curvrank._plugin  import *"\
 && /virtualenv/env3/bin/python -c "import wbia_finfindr;   from wbia_finfindr._plugin  import *" \
 && /virtualenv/env3/bin/python -c "import wbia_kaggle7;    from wbia_kaggle7._plugin   import *" \
 && /virtualenv/env3/bin/python -c "import wbia_deepsense;  from wbia_deepsense._plugin import *"

##########################################################################################
FROM wildme/wbia-base:latest as org.wildme.wbia.install

LABEL autoheal=true

ENV HOST_USER=root

ENV HOST_UID=0

ENV AWS_ACCESS_KEY_ID

ENV AWS_SECRET_ACCESS_KEY

COPY --from=org.wildme.wbia.build /virtualenv /virtualenv

COPY --from=org.wildme.wbia.build /wbia /wbia

# Add theano configuration file
ADD ./_config/theanorc /root/.theanorc

# Add embed script for python development
ADD ./_config/embed.sh /bin/embed

# Add embed script for python development
ADD ./_config/shell.sh /bin/shell

# Add Python health check
ADD ./_config/healthcheck.py /bin/healthcheck.py

# Add entrypoint
ADD ./_config/entrypoint.sh /bin/entrypoint

# Specify database directory
RUN set -x \
 && mkdir -p /data \
 && /virtualenv/env3/bin/python -m wbia.dev --set-workdir /data

WORKDIR /wbia

# Port for the web server
EXPOSE 5000

HEALTHCHECK --interval=2m --timeout=2m --retries=10 --start-period=1h \
  CMD /virtualenv/env3/bin/python /bin/healthcheck.py

STOPSIGNAL SIGTERM

ENTRYPOINT ["/bin/entrypoint", "/virtualenv/env3/bin/python -m wbia.dev --dbdir /data/db --logdir /data/db/logs --web --containerized --production"]

##########################################################################################
FROM org.wildme.wbia.install as org.wildme.wbia.configure

CMD []
