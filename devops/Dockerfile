FROM wildme/wbia-provision:latest as org.wildme.wbia.build

MAINTAINER Wild Me <dev@wildme.org>

# Remove all git association from the code
RUN set -ex \
 && git config --global user.email "dev@wildme.org" \
 && git config --global user.name "Wild Me" \
 && find /wbia/wbia* -name '.git' -type d -print0 | xargs -0 -i /bin/bash -c 'cd {} && cd .. && echo $(pwd) && (git stash && git pull && git stash pop || git reset --hard origin/develop)' \
 && cd /wbia/wbia-plugin-curvrank/wbia_curvrank \
 && git stash && git pull && git stash pop || git reset --hard origin/develop \
 && cd /wbia/wbia-plugin-kaggle7/wbia_kaggle7 \
 && git stash && git pull && git stash pop || git reset --hard origin/develop \
 && find /wbia -name '.git' -type d -print0 | xargs -0 rm -rf

# Run smoke tests
# WBIA toolkit repositories
RUN set -ex \
 && /virtualenv/env3/bin/python -c "import utool;           from utool.__main__ import main;            main()" \
 && /virtualenv/env3/bin/python -c "import vtool;           from vtool.__main__ import main;            main()"

# WBIA third-party toolkit repositories 
RUN set -ex \
 && /virtualenv/env3/bin/python -c "import pyhesaff;        from pyhesaff.__main__ import main;         main()" \
 && /virtualenv/env3/bin/python -c "import pyflann;         from pyflann.__main__ import main;          main()" \
 && /virtualenv/env3/bin/python -c "import pydarknet;       from pydarknet.__main__ import main;        main()" \
 && /virtualenv/env3/bin/python -c "import pyrf;            from pyrf.__main__ import main;             main()" \
 && /virtualenv/env3/bin/python -c "import brambox;         from brambox import __version__, __file__;  print('Brambox: %r, %r' % (__version__, __file__, ))" \
 && /virtualenv/env3/bin/python -c "import lightnet;        from lightnet import __version__, __file__; print('Lightnet: %r, %r' % (__version__, __file__, ))"

# first-party WBIA plug-in repositories
RUN set -ex \
 && /virtualenv/env3/bin/python -c "import wbia_pie;        from wbia_pie.__main__ import main;         main()"

RUN set -ex \
 && /virtualenv/env3/bin/python -c "import wbia_flukematch; from wbia_flukematch.plugin import *" \
 && /virtualenv/env3/bin/python -c "import wbia_curvrank;   from wbia_curvrank._plugin import *"\
 && /virtualenv/env3/bin/python -c "import wbia_finfindr;   from wbia_finfindr._plugin import *" \
 && /virtualenv/env3/bin/python -c "import wbia_kaggle7;    from wbia_kaggle7._plugin import *" \
 && /virtualenv/env3/bin/python -c "import wbia_deepsense;  from wbia_deepsense._plugin import *"

# WBIA repository
RUN set -ex \
 && /virtualenv/env3/bin/python -c "import wbia_cnn;        from wbia_cnn.__main__ import wbia_cnn_main as main; main()" \
 && /virtualenv/env3/bin/python -c "import wbia;            from wbia.__main__ import smoke_test as main;        main()"

##########################################################################################
FROM wildme/wbia-base:latest as org.wildme.wbia.install

MAINTAINER Wild Me <dev@wildme.org>

COPY --from=org.wildme.wbia.build /virtualenv /virtualenv

COPY --from=org.wildme.wbia.build /wbia /wbia

# Add theano configuration file
ADD ./_config/theanorc /root/.theanorc

# Add embed script for python development
ADD ./_config/embed.sh /bin/embed

# Add Python health check
ADD ./_config/healthcheck.py /bin/healthcheck.py

RUN set -x \
    && mkdir -p /data\
    && chown wbia:wbia /data \
    && chmod 755 /data

WORKDIR /wbia

RUN /virtualenv/env3/bin/python -m wbia.dev --set-workdir /data

##########################################################################################
FROM org.wildme.wbia.install as org.wildme.wbia.deploy

MAINTAINER Wild Me <dev@wildme.org>

LABEL autoheal=true

COPY ./_config/entrypoint.sh /bin/entrypoint

ENTRYPOINT ["/bin/entrypoint", "/virtualenv/env3/bin/python -m wbia.dev --dbdir /data/db --logdir /data/logs --web --containerized --production"]

# Port for the web server
EXPOSE 5000

HEALTHCHECK --interval=2m --timeout=2m --retries=10 --start-period=1h \
  CMD /virtualenv/env3/bin/python /bin/healthcheck.py

STOPSIGNAL SIGTERM

##########################################################################################
FROM org.wildme.wbia.deploy as org.wildme.wbia.configure

MAINTAINER Wild Me <dev@wildme.org>

CMD []
