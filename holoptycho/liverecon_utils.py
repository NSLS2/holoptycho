import configparser

class header_param(object):
    pass

def parse_scan_header(filename):
    config = configparser.ConfigParser(inline_comment_prefixes=('#',))
    try:
        config.read(filename)
        p = header_param()

        p.scan_num          = config.getint('scan', 'scan_num')
        p.x_num             = config.getint('scan', 'x_num')
        p.y_num             = config.getint('scan', 'y_num')
        p.nz                = config.getint('scan', 'nz')
        p.det_roix0         = config.getint('scan', 'det_roix_start')
        p.det_roiy0         = config.getint('scan', 'det_roiy_start')

        p.x_range           = config.getfloat('scan', 'x_range')
        p.y_range           = config.getfloat('scan', 'y_range')
        p.angle             = config.getfloat('scan', 'angle' , fallback=0)

        p.x_motor           = config.get('scan', 'xmotor')
        p.y_motor           = config.get('scan', 'ymotor')

        return p
        
    except Exception as err:
        # print(err)
        return None
