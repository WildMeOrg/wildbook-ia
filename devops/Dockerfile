FROM wildme/wbia-provision:latest as org.wildme.wbia.build

MAINTAINER Wild Me <dev@wildme.org>

# Remove all git association from the code
RUN find /wbia -name '.git' -type d -print0 | xargs -0 rm -rf

RUN set -ex \
 && /virtualenv/env3/bin/pip uninstall -y \
        opencv-python \
        opencv-python-headless \
 && /virtualenv/env3/bin/pip uninstall -y \
        dtool-ibeis \
        guitool-ibeis \
        plottool-ibeis \
        utool-ibeis \
        vtool-ibeis \
 && /virtualenv/env3/bin/pip uninstall -y \
        dtool \
        guitool \
        plottool \
        utool \
        vtool \
 && /virtualenv/env3/bin/pip uninstall -y \
        tensorflow \
        tensorflow-gpu \
        tensorflow-estimator \
        tensorboard \
        tensorboard-plugin-wit \
        keras

RUN set -ex \
 && /virtualenv/env3/bin/pip install \
        # opencv-python==3.4.11.41 \
        tensorflow-gpu==1.15 \
        keras==2.2.5

# Run smoke test
RUN set -ex \
 # WBIA repository
 && /virtualenv/env3/bin/python -c "import wbia            as check; from check.__main__ import main; main()" \
 # WBIA toolkit repositories      
 && /virtualenv/env3/bin/python -c "import utool           as check; from check.__main__ import main; main()" \
 && /virtualenv/env3/bin/python -c "import vtool           as check; from check.__main__ import main; main()" \
 # WBIA third-party toolkit repositories      
 && /virtualenv/env3/bin/python -c "import pyhesaff        as check; from check.__main__ import main; main()" \
 && /virtualenv/env3/bin/python -c "import pyflann         as check; from check.__main__ import main; main()" \
 && /virtualenv/env3/bin/python -c "import pydarknet       as check; from check.__main__ import main; main()" \
 && /virtualenv/env3/bin/python -c "import pyrf            as check; from check.__main__ import main; main()" \
 && /virtualenv/env3/bin/python -c "import brambox         as check; from check.__main__ import main; main()" \
 && /virtualenv/env3/bin/python -c "import lightnet        as check; from check.__main__ import main; main()" \
 # first-party WBIA plug-in repositories
 && /virtualenv/env3/bin/python -c "import wbia_cnn        as check; from check.__main__ import main; main()" \
 && /virtualenv/env3/bin/python -c "import wbia_flukematch as check; from check.__main__ import main; main()" \
 && /virtualenv/env3/bin/python -c "import wbia_finfindr   as check; from check.__main__ import main; main()" \
 && /virtualenv/env3/bin/python -c "import wbia_deepsense  as check; from check.__main__ import main; main()" \
 && /virtualenv/env3/bin/python -c "import wbia_pie        as check; from check.__main__ import main; main()" \
 # third-party WBIA plug-in repositories
 && /virtualenv/env3/bin/python -c "import wbia_curvrank   as check; from check.__main__ import main; main()" \
 && /virtualenv/env3/bin/python -c "import wbia_kaggle7    as check; from check.__main__ import main; main()"

##########################################################################################
FROM wildme/wbia-base:latest as org.wildme.wbia.install

MAINTAINER Wild Me <dev@wildme.org>

COPY --from=org.wildme.wbia.build /virtualenv /virtualenv

COPY --from=org.wildme.wbia.build /wbia /wbia

# Add theano configuration file
ADD ./provision/_config/theanorc /root/.theanorc

# Add embed script for python development
ADD ./provision/_config/embed.sh /bin/embed

# Add Python health check
ADD ./provision/_config/healthcheck.py /bin/healthcheck.py

RUN mkdir /data

VOLUME /data

RUN /virtualenv/env3/bin/python -m wbia --set-workdir /data

##########################################################################################
FROM org.wildme.wbia.install as org.wildme.wbia.deploy

MAINTAINER Wild Me <dev@wildme.org>

LABEL autoheal=true

ENTRYPOINT ["/virtualenv/env3/bin/python", "-m", "wbia" "--dbdir", "/data/db", "--logdir", "/data/logs", "--web", "--port", "5000", "--web-deterministic-ports", "--cpudark", "--containerized", "--production"]

# Ports for the frontend web server
EXPOSE 5000

HEALTHCHECK --interval=2m --timeout=2m --retries=10 --start-period=1h \
  CMD /virtualenv/env3/bin/python /bin/healthcheck.py

STOPSIGNAL SIGTERM

##########################################################################################
FROM org.wildme.wbia.deploy as org.wildme.wbia.configure

MAINTAINER Wild Me <dev@wildme.org>

CMD []
