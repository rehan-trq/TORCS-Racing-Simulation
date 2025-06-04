class MsgParser:
    '''
    A parser for received UDP messages and building UDP messages
    '''
    def __init__(self):
        '''Constructor'''
        pass  # Added 'pass' to avoid syntax errors

    def parse(self, str_sensors):
        '''Return a dictionary with tags and values from the UDP message'''
        sensors = {}
        
        b_open = str_sensors.find('(')
        
        while b_open >= 0:
            b_close = str_sensors.find(')', b_open)
            if b_close >= 0:
                substr = str_sensors[b_open + 1: b_close]
                items = substr.split()
                if len(items) < 2:
                    print("Problem parsing substring:", substr)
                else:
                    value = [items[i] for i in range(1, len(items))]
                    sensors[items[0]] = value
                b_open = str_sensors.find('(', b_close)
            else:
                print("Problem parsing sensor string:", str_sensors)
                return None
        
        return sensors
    
    def stringify(self, dictionary):
        '''Build a UDP message from a dictionary'''
        msg = ''
        
        for key, value in dictionary.items():
            if value is not None and value[0] is not None:
                msg += f"({key}"
                for val in value:
                    msg += f" {val}"
                msg += ")"
        
        return msg
