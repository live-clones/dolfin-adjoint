# Builds a Docker image with dolfin-adjoint stable version built from
# git sources. The image is at:
#
#    https://quay.io/repository/dolfinadjoint/dev-dolfin-adjoint
#
# Authors:
# Simon W. Funke <simon@simula.no>
# Jack S. Hale <jack.hale@uni.lu>

FROM quay.io/fenicsproject/dev-env:latest
MAINTAINER Simon W. Funke <simon@simula.no>

USER root
RUN apt-get -qq update && \
    apt-get -y install python-scipy emacs-nox nano gdb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER fenics
ENV FENICS_BUILD_TYPE=Release
RUN bash -c "source ~/fenics.env.conf && \
             PIP_NO_CACHE_DIR=off fenics-update"
ENV DOLFIN_ADJOINT_VERSION master
ENV IPOPT_VER=3.12.6
COPY dolfin-adjoint.conf $FENICS_HOME/dolfin-adjoint.conf
RUN /bin/bash -l -c "source $FENICS_HOME/dolfin-adjoint.conf && \
                     update_libadjoint && \
                     update_dolfin-adjoint && \
                     update_moola && \
                     update_ipopt && \
                     update_pyipopt && \
                     rm -rf $FENICS_HOME/build/src"

COPY WELCOME $FENICS_HOME/WELCOME
RUN echo "source $FENICS_HOME/dolfin-adjoint.conf" >> $FENICS_HOME/.bash_profile

USER root
