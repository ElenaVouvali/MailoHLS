"""Versioned compiler-evidence vocabulary shared by export and alignment."""
TYPES = ('memory_root', 'memory_ssa', 'loop', 'control', 'access', 'arithmetic',
         'carried_value', 'memory_access', 'loop_carried', 'depends_on',
         'loop_child', 'operand', 'defines')
STATUS = ('proven', 'possible', 'unknown', 'not_established', 'not_applicable')
KNOWN = ('class', 'proven', 'shape', 'trip_count', 'affine_map', 'access_role',
         'operation_family', 'loop_carried', 'dependence_kind')
