# This is a stub for the steps required to create a new release

echo "Creating release tar ball"
PACKAGE=dolfin-adjoint
VERSION=2016.1.0
ARCHIVE=$PACKAGE-$VERSION.tar.gz
git archive --prefix=$PACKAGE-$VERSION/ -o $ARCHIVE $PACKAGE-$VERSION

