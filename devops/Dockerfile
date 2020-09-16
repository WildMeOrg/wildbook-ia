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
 # WBIA toolkit repositories
 && /virtualenv/env3/bin/python -c "import utool;           from utool.__main__ import main;            main()" \
 && /virtualenv/env3/bin/python -c "import vtool;           from vtool.__main__ import main;            main()" \
 # WBIA third-party toolkit repositories 
 && /virtualenv/env3/bin/python -c "import pyhesaff;        from pyhesaff.__main__ import main;         main()" \
 && /virtualenv/env3/bin/python -c "import pyflann;         from pyflann.__main__ import main;          main()" \
 && /virtualenv/env3/bin/python -c "import pydarknet;       from pydarknet.__main__ import main;        main()" \
 && /virtualenv/env3/bin/python -c "import pyrf;            from pyrf.__main__ import main;             main()" \
 && /virtualenv/env3/bin/python -c "import brambox;         from brambox import __version__, __file__;  print('Brambox: %r, %r' % (__version__, __file__, ))" \
 && /virtualenv/env3/bin/python -c "import lightnet;        from lightnet import __version__, __file__; print('Lightnet: %r, %r' % (__version__, __file__, ))" \
 # first-party WBIA plug-in repositories
 && /virtualenv/env3/bin/python -c "import wbia_cnn;        from wbia_cnn.__main__ import main;         main()" \
 && /virtualenv/env3/bin/python -c "import wbia_flukematch; from wbia_flukematch.plugin import *" \
 && /virtualenv/env3/bin/python -c "import wbia_finfindr;   from wbia_finfindr._plugin import *" \
 && /virtualenv/env3/bin/python -c "import wbia_deepsense;  from wbia_deepsense.__main__ import main;   main()" \
 && /virtualenv/env3/bin/python -c "import wbia_pie;        from wbia_pie.__main__ import main;         main()" \
 # third-party WBIA plug-in repositories
 && /virtualenv/env3/bin/python -c "import wbia_curvrank;   from wbia_curvrank.__main__ import main;    main()" \
 && /virtualenv/env3/bin/python -c "import wbia_kaggle7;    from wbia_kaggle7.__main__ import main;     main()" \
 # WBIA repository
 && /virtualenv/env3/bin/python -c "import wbia;            from wbia.__main__ import main;             main()" \


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
