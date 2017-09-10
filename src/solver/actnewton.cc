#include <picasso/actnewton.h>
#include <dmlc/parameter.h>

namespace picasso {
namespace solver {
DMLC_REGISTRY_FILE_TAG(solver);

struct SpLinearModelParam: public dmlc::Parameter<SpLinearModelParam> {
  // feature dimension
  unsigned dim_feature;

  SpLinearModelParam() {
    std::memset(this, 0, sizeof(SpLinearModelParam));
  }

  DMLC_DECLARE
};

}
}